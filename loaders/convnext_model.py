# ------------------------------------------------------------------------
# LocalizationLabelNoise
# Copyright (c) 2023 Jonas Uhrig, Jeethesh Pai Umesh. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ConvNeXt (https://github.com/facebookresearch/ConvNeXt):
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# Config for ConvNext models are as follows:
# ConvNext Tiny := ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
# ConvNext Small := ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
# ConvNext Large := ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
# ConvNext xLarge := ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
# ConvNext Base := ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0, layer_scale_init_value=1e-6
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def normless_forward_feature(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.normless_forward_feature(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BBoxHead(nn.Module):
    """Creates Box Head on which regresses to exact bounding box coordinates given the
    pooled ROI's from the false bounding box suggestions. This currently uses TWOMLPHead
    in pytorch for last FC layers
    Args:
        in_channels (int): Number of channels of the input feature map provided
        kernel_size (int): size of the kernel to be used for filter
        num_layers (int):
    """

    def __init__(self, in_channels=768, kernel_size=3, num_layers=3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=kernel_size, padding="same")
        self.convLayers = nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(1, num_layers):
            module = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, padding="same", kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(num_features=512),
            )
            self.convLayers.append(module)
        self.fcLayers = nn.Sequential(
            nn.Linear(32768, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 1024), nn.ReLU(inplace=True)  # 512 x 8 x 8
        )
        self.regHead = nn.Linear(1024, 4)

    def forward(self, x):
        """calls forward function nn.Module

        Args:
            x (_type_): Feature map from backbone with in_channel number of channels
        """
        x = self.conv1(x)
        for i in range(self.num_layers - 1):
            x = self.convLayers[i](x)

        x = torch.flatten(x, start_dim=1)

        x = self.fcLayers(x)
        norm = torch.norm(x, dim=-1, keepdim=True)
        x = torch.div(x, norm)
        x = self.regHead(x)

        return x


class ConvNextFPN(nn.Module):
    def __init__(
        self, convnext_weights: str, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], FPN_channels=256, freeze=False
    ) -> None:
        super().__init__()

        self.convnext = ConvNeXt(depths=depths, dims=dims)
        weights = torch.load(convnext_weights)["model"]
        print(f"Loading the weights from {convnext_weights} .. skipping unmatched layers")
        self.convnext.load_state_dict(weights, strict=False)
        if freeze:
            for _, params in self.convnext.named_parameters():
                params.requires_grad_(False)
        self.dims = dims
        self.depths = depths
        self.FPN_channels = FPN_channels
        self.smootherModule = nn.ModuleList()
        self.lateral_module = nn.ModuleList()
        for num in range(len(dims)):
            conv = nn.Conv2d(in_channels=dims[num], out_channels=FPN_channels, kernel_size=1, padding="same", bias=True)
            self.lateral_module.append(conv)
            conv_smooth = nn.Conv2d(
                in_channels=FPN_channels, out_channels=FPN_channels, kernel_size=3, padding="same", bias=True
            )
            self.smootherModule.append(conv_smooth)
        del self.smootherModule[-1]  # no need for smoothing last conv layer.

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain("conv2d"))
            nn.init.zeros_(m.bias.data)

    def initialize(self):
        for i in range(len(self.dims) - 1):
            self.init_weights(self.lateral_module[i])
            self.init_weights(self.smootherModule[i])  # len(smoother) = len(lateral) - 1
        self.init_weights(self.lateral_module[-1])

    def forward(self, x):
        laterals = []
        features = []
        for i in range(4):
            x = self.convnext.downsample_layers[i](x)
            x = self.convnext.stages[i](x)
            laterals.append(self.lateral_module[i](x))  # all features should have same channels - laterals
        features.append(laterals[-1])
        for i in range(len(laterals) - 2, -1, -1):
            merge_levels = laterals[i] + F.interpolate(
                laterals[i + 1], scale_factor=2, mode="bilinear"
            )  # interepolate and add subsequent layers
            smooth_conv = self.smootherModule[i](merge_levels)  # anti-aliasing for interpolated features
            features.append(smooth_conv)
        features = features[::-1]  # arrange features according to bottom up pyramid not top-down pyramid
        return features

    def feat_spatial_scale(self):
        return [2 ** (2 + i) for i in range(len(self.dims))]
