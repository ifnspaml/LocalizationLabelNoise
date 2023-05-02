from typing import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from loaders.convnext_model import ConvNextFPN
from utils.utils import UBBR_unreplicate_pred_boxes, decode_pred_bbox_xyxy_xyxy


class ConvFCBBoxHead(nn.Module):
    def __init__(
        self,
        num_conv: int,
        num_fc: int,
        in_channels: int,
        conv_out_channels: int,
        roi_feature_size: int,
        fc_out_features: int,
        std_head=True,
        class_head=False,
        num_classes=20,
    ) -> None:
        """Generates chain of Convolutional blocks with Batchnormalization and activation given
        on the top of extracted feature map. This block is meant to follow the ROI Align / ROI Pool layer
        and is designed to produce bbox coordinate offsets (parameterized). This can also tuned to produce
        uncertainty by using the argument "std_head=True" for training with KL Loss

        Args:
            num_conv (int): Number of Convolutional layers after ROI Align
            num_fc (int): number of fully connected layers after flattening.
                        Arch - ROI Align -> CONV -> Flatten -> FC -> FC(4)
            in_channels (int): Number of channel of the ROI Align layer
            conv_out_channels (int): Number of channels for Convolution layer
            roi_feature_size (int): feature map dimension of ROI Align layer
            fc_out_features (int): Number of units of the Fully connected layers
            std_head (bool, optional): Whether to produce extra variance for KL Loss. Defaults to True.
            class_head (bool, optional): Whether to predict a class label. Defaults to False.
            num_classes (int, optional): Number of classes for which the class_head is to be trained. Defaults to 20.
        """
        super().__init__()
        self.num_conv = num_conv
        self.num_fc = num_fc
        self.in_channels = in_channels
        self.out_channels = conv_out_channels
        self.roi_feature_size = roi_feature_size
        self.fc_out_feature = fc_out_features
        self.std_head = std_head
        self.class_head = class_head
        self.num_classes = num_classes
        self.ConvList = nn.ModuleList()
        self.FCList = nn.ModuleList()

        for num in range(self.num_conv):
            if num == 0:
                conv = nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding="same",
                    bias=False,
                )
                BN = nn.BatchNorm2d(self.out_channels)
            else:
                conv = nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding="same",
                    bias=False,
                )
                BN = nn.BatchNorm2d(self.out_channels)
            self.ConvList.append(conv)
            self.ConvList.append(BN)
        for num in range(self.num_fc):
            if num == 0:
                fc_layer = nn.Linear(
                    self.out_channels * self.roi_feature_size * self.roi_feature_size,
                    out_features=self.fc_out_feature,
                )
            else:
                fc_layer = nn.Linear(self.fc_out_feature, self.fc_out_feature)

            if num == self.num_fc - 1:  # check if this is last layer
                fc_layer = nn.Linear(self.fc_out_feature, 4)
                if std_head:
                    self.fc_std = nn.Linear(
                        self.fc_out_feature, out_features=4
                    )  # include std along with mean / estimated box coordinate
                if self.class_head:
                    self.fc_class = nn.Linear(
                        self.fc_out_feature, out_features=self.num_classes
                    )  # include a class head
            if self.num_fc == 1:
                fc_layer = nn.Linear(
                    self.out_channels * self.roi_feature_size * self.roi_feature_size,
                    out_features=4,
                )
                if std_head:
                    self.fc_std = nn.Linear(
                        self.out_channels * self.roi_feature_size * self.roi_feature_size,
                        out_features=4,
                    )  # include std along with mean / estimated box coordinate
                if self.class_head:
                    self.fc_class = nn.Linear(
                        self.out_channels * self.roi_feature_size * self.roi_feature_size,
                        out_features=self.num_classes,
                    )
            self.FCList.append(fc_layer)

    def init_weights(self, m: nn.Module):
        """Initializes the weights of the ConvFCBBoxHead module

        Args:
            m (nn.Module): layers of ConvFCBBoxHead
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.001)  # FC Layers overshoot if the variance value is more than 0.001
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        else:
            AssertionError(TypeError)

    def initialize(self):
        for num in range(self.num_conv):
            self.init_weights(self.ConvList[2 * num])
            self.init_weights(self.ConvList[2 * num + 1])
        for num in range(self.num_fc):
            self.init_weights(self.FCList[num])
        if self.std_head:
            nn.init.normal_(self.fc_std.weight.data, 0, 1e-4)
            nn.init.zeros_(self.fc_std.bias.data)
        if self.class_head:
            nn.init.xavier_uniform_(self.fc_class.weight.data)
            nn.init.zeros_(self.fc_class.bias.data)

    def forward(self, x):
        """feeds forward the roi proposal on the BBOX head architecture

        Args:
            x (torch.Tensor): Pooled features
        """

        for num in range(self.num_conv):
            x = self.ConvList[2 * num](x)
            x = F.relu(x)
            x = self.ConvList[2 * num + 1](x)

        x = torch.flatten(x, 1)

        for num in range(self.num_fc - 1):
            x = self.FCList[num](x)
            x = F.relu(x)
        x_reg = x
        x_cls = x
        x_bbox = self.FCList[-1](x_reg)
        if self.class_head:
            x_cls = self.fc_class(x_cls)
            if self.std_head:
                x_std = self.fc_std(x_reg)
                return {"reg": (x_bbox, x_std), "class": (x, x_cls)}
            else:
                return (
                    x_bbox,
                    x_cls,
                    x,
                )  # predicts the 4 coordinates, class label of the ROI given
            # and feature map just before FC(4) layer for cosine loss
        if self.std_head:
            x_std = self.fc_std(x_reg)
            return (
                x_bbox,
                x_std,
            )  # predicts the 4 coordinates and 4 variance value associated with them
        return x_bbox


class IterModel(nn.Module):
    def __init__(
        self,
        weights: str,
        num_stages: int,
        num_fc: int,
        num_conv: int,
        roi_feature_size=8,
        fc_out_features=1024,
        roi_channels=256,
        freeze=True,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        input_size=(3, 512, 512),
        std_head=False,
        class_head=False,
        init=True,
        num_classes=20,
    ) -> None:
        super().__init__()
        """Creates the Iterative model as explained in the presentation. The iteerative model has same head for every iteration. 
        The output produced includes parameterized as well as de-parameterized outputs of all iterations. The output is dict containing keywords 
        "preds" and "param_pred" meaning the de-parameterized coordinates and parameterized coordinates
        """

        self.num_stages = num_stages
        self.backbone = ConvNextFPN(convnext_weights=weights, depths=depths, dims=dims, freeze=freeze)
        self.input_size = input_size
        self.roi_feature_size = roi_feature_size
        self.roi_channels = roi_channels
        self.dims = dims
        self.scales = self.backbone.feat_spatial_scale()
        self.init = init
        self.std_head = std_head
        self.class_head = class_head
        self.num_classes = num_classes
        self.feat_names = ["feat1", "feat2", "feat3", "feat4"]
        self.head = ConvFCBBoxHead(
            num_conv=num_conv,
            num_fc=num_fc,
            in_channels=roi_channels,
            conv_out_channels=256,
            roi_feature_size=self.roi_feature_size,
            fc_out_features=fc_out_features,
            std_head=self.std_head,
            class_head=self.class_head,
            num_classes=self.num_classes,
        )
        if self.init:
            self.init_weights()
        self.MultiScaleRoIAlign = MultiScaleRoIAlign(
            ["feat1", "feat2", "feat3", "feat4"],
            output_size=self.roi_feature_size,
            sampling_ratio=-1,
            canonical_scale=224,
            canonical_level=4,
        )

    def init_weights(self):
        self.head.initialize()

    def forward(self, image: torch.Tensor, proposal_boxes: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.backbone(image)
        feature_pyramid = OrderedDict(list(zip(self.feat_names, feature_pyramid)))
        roi_proposals = proposal_boxes
        result_dict = {"param_preds": [], "preds": []}
        for _ in range(self.num_stages):
            final_roi = self.MultiScaleRoIAlign(
                feature_pyramid,
                proposal_boxes,
                image_shapes=[(self.input_size[1], self.input_size[2])],
            )
            preds = self.head(final_roi)
            act_preds = decode_pred_bbox_xyxy_xyxy(proposal_boxes, preds, (self.input_size[1], self.input_size[2]))
            result_dict["param_preds"].append(preds)
            result_dict["preds"].append(act_preds)
            act_preds = UBBR_unreplicate_pred_boxes(act_preds, roi_proposals)
            proposal_boxes = act_preds
        return result_dict


class CascadeHead(nn.Module):
    def __init__(
        self,
        weights: str,
        num_stages: int,
        num_fc: int,
        num_conv: int,
        roi_feature_size=8,
        fc_out_features=1024,
        roi_channels=256,
        freeze=True,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        input_size=(3, 512, 512),
        std_head=False,
        class_head=False,
        init=True,
        num_classes=20,
    ) -> None:
        super().__init__()
        """Implements the architecture of cascade head shown in the presentation. For every iteration there is a head
        taking care of the iterative refinement. outputs same as IterModel above
        """
        self.num_stages = num_stages
        self.backbone = ConvNextFPN(
            convnext_weights=weights, depths=depths, dims=dims, freeze=freeze
        )  # convnext backbone pretrained on ImageNet
        self.input_size = input_size
        self.roi_feature_size = roi_feature_size
        self.roi_channels = roi_channels
        self.dims = dims
        self.head = nn.ModuleList()
        self.scales = self.backbone.feat_spatial_scale()
        self.init = init
        self.std_head = std_head
        self.class_head = class_head
        self.num_classes = num_classes
        self.feat_names = ["feat1", "feat2", "feat3", "feat4"]
        for _ in range(self.num_stages):
            boxhead = ConvFCBBoxHead(
                num_conv=num_conv,
                num_fc=num_fc,
                in_channels=roi_channels,
                conv_out_channels=256,
                roi_feature_size=self.roi_feature_size,
                fc_out_features=fc_out_features,
                std_head=self.std_head,
                class_head=self.class_head,
                num_classes=self.num_classes,
            )
            self.head.append(boxhead)
        if self.init:
            self.init_weights()
        self.MultiScaleRoIAlign = MultiScaleRoIAlign(
            ["feat1", "feat2", "feat3", "feat4"],
            output_size=self.roi_feature_size,
            sampling_ratio=-1,
            canonical_scale=224,
            canonical_level=4,
        )

    def init_weights(self):
        for num in range(self.num_stages):
            self.head[num].initialize()

    def forward(self, image: torch.Tensor, proposal_boxes: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.backbone(image)
        feature_pyramid = OrderedDict(list(zip(self.feat_names, feature_pyramid)))
        roi_proposals = proposal_boxes
        result_dict = {"param_preds": [], "preds": []}
        for num in range(self.num_stages):
            final_roi = self.MultiScaleRoIAlign(
                feature_pyramid,
                proposal_boxes,
                image_shapes=[(self.input_size[1], self.input_size[2])],
            )
            preds = self.head[num](final_roi)
            act_preds = decode_pred_bbox_xyxy_xyxy(proposal_boxes, preds, (self.input_size[1], self.input_size[2]))
            result_dict["param_preds"].append(preds)
            result_dict["preds"].append(act_preds)
            act_preds = UBBR_unreplicate_pred_boxes(act_preds, roi_proposals)
            proposal_boxes = act_preds
        return result_dict


class FasterRCNN(nn.Module):
    def __init__(
        self,
        weights: str,
        num_fc: int,
        num_conv: int,
        roi_feature_size=7,
        fc_out_features=1024,
        roi_channels=256,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        input_size=(3, 512, 512),
        init=True,
        freeze=True,
        std_head=False,
        class_head=False,
        num_classes=20,
    ) -> None:
        """Creates model responsible for non-iterative refinement as per the thesis. This class can
        produce coordinate outputs (default), Variance (using std_head=True) for KL Loss,
        Class output and feature map output (using class_head=True) for cosine loss and cross entropy loss

        Args:
            weights (str): path to .pth file having ConvNext Backbone of dimensions specified in "arguments: depths, dims"
            num_conv (int): Number of Convolutional layers after ROI Align
            num_fc (int): number of fully connected layers after flattening.
                        Arch - ROI Align -> CONV -> Flatten -> FC -> FC(num_classes)
            roi_channels (int): Number of channels of ROI Aligned output
            roi_feature_size (int): feature map dimension of ROI Align layer
            fc_out_features (int): Number of units of the Fully connected layers
            depths (list, optional): Architecture used in ConvNext. Defaults to ConvNext-T [3, 3, 9, 3].
            dims (list, optional): Number of channel in the Conv_blocks of Convnext. Defaults to Convnext-T[96, 192, 384, 768].
            input_size (tuple, optional): Image input dimensions (N, H, W). Defaults to (3, 512, 512).
            init (bool, optional): Whether to initialize the backbone or not. Defaults to True.
            freeze (bool, optional): Whether to freeze the backbone or not. Defaults to True.
            std_head (bool, optional): If True produces an extra variance for every predicted coordinate. Defaults to False.
            class_head (bool, optional): Predicts class label. Defaults to False.
            num_classes (int, optional): Number of classes of dataset to be trained excluding background. Defaults to 20.

        Return:
        Outputs dictionary depends on std_head, class_head boolean values. See the output of ConvFCBBoxHead and ConvFCClassHead for details
        """
        super().__init__()
        self.backbone = ConvNextFPN(convnext_weights=weights, depths=depths, dims=dims, freeze=freeze)
        self.input_size = input_size
        self.roi_feature_size = roi_feature_size
        self.roi_channels = roi_channels
        self.dims = dims
        self.scales = self.backbone.feat_spatial_scale()
        self.init = init
        self.std_head = std_head
        self.class_head = class_head
        self.num_classes = num_classes
        self.head = ConvFCBBoxHead(
            num_conv=num_conv,
            num_fc=num_fc,
            in_channels=roi_channels,
            conv_out_channels=256,
            roi_feature_size=self.roi_feature_size,
            fc_out_features=fc_out_features,
            std_head=std_head,
            class_head=self.class_head,
            num_classes=self.num_classes,
        )
        if self.init:
            self.init_weights()
        self.MultiScaleRoIAlign = MultiScaleRoIAlign(
            ["feat1", "feat2", "feat3", "feat4"],
            output_size=self.roi_feature_size,
            sampling_ratio=-1,
            canonical_scale=224,
            canonical_level=4,
        )
        self.feat_names = ["feat1", "feat2", "feat3", "feat4"]

    def init_weights(self):
        self.head.initialize()
        self.backbone.initialize()

    def forward(self, image: torch.Tensor, proposal_boxes: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.backbone(image)

        feature_pyramid = OrderedDict(list(zip(self.feat_names, feature_pyramid)))

        final_roi = self.MultiScaleRoIAlign(
            feature_pyramid,
            proposal_boxes,
            image_shapes=[(self.input_size[1], self.input_size[2])],
        )

        if torch.any(torch.isnan(final_roi)):
            print("Nan in roi")
        preds = self.head(final_roi)
        return preds
