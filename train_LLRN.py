import argparse
import os
import pathlib
import shutil
import warnings

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from loaders.cascade_roi import CascadeHead, FasterRCNN, IterModel
from loaders.dataloader import VOCAnnotationTransform, VOCLoader
from utils.eval_utils import box_iou
from utils.misc_utils import add_dict, make_histogram, print_dict
from utils.utils import (
    IoU_encoded_L1_loss,
    IoUL1_loss,
    UBBR_collate_fn,
    UBBR_replicate_target_boxes,
    decode_pred_bbox_xyxy_xyxy,
    encode_xyxy_xyxy,
)

warnings.simplefilter("ignore")


def get_empty_epoch_entries(num_stages):
    iou_dist_dict = {
        "less_30": 0,
        "30_40": 0,
        "40_50": 0,
        "50_60": 0,
        "60_70": 0,
        "70_80": 0,
        "80_90": 0,
        "greater_90": 0,
    }
    mean_stage_iou = [0] * num_stages
    mean_stage_loss = [0] * num_stages
    epoch_loss = 0
    return iou_dist_dict, mean_stage_iou, mean_stage_loss, epoch_loss


def update_progress_bar(
    progress_bar,
    data_split,
    mean_stage_iou,
    mean_stage_loss,
    num_stages,
    batch_idx,
    loss,
    epoch_loss,
    epoch,
    roi_iou,
):
    iou_string = ", ".join([f"IoU stage {stage}: {np.round(mean_stage_iou[stage], 2)}" for stage in range(num_stages)])
    loss_string = ", ".join(
        [f"loss stage {stage}: {np.round(mean_stage_loss[stage], 2)}" for stage in range(num_stages)]
    )
    epoch_loss = (epoch_loss + loss.item()) / (1 if batch_idx == 0 else 2)
    progress_bar.set_description(
        f"{data_split} @ epoch: {epoch}: Total Loss: {epoch_loss:.4f} "
        + loss_string
        + f" , RoI IoU: {np.round(roi_iou, 2)}, "
        + iou_string,
        refresh=True,
    )
    return epoch_loss


def forward(
    data,
    device,
    model,
    num_stages,
    model_type,
    img_size,
    mean_stage_iou,
    mean_stage_loss,
    batch_idx,
    iou_dist_dict,
):
    image = data[0].to(device)

    target_dict = data[1]
    target = target_dict["boxes"]
    noisy_boxes = target_dict["falseBoxes"]
    roi_noisy_boxes = [torch.cat([*box], dim=0) for box in noisy_boxes]
    roi_noisy_boxes = [box.to(device) for box in roi_noisy_boxes]
    pred = model(image, roi_noisy_boxes)
    target_expanded = UBBR_replicate_target_boxes(target, noisy_boxes).to(device)
    target_encoded = encode_xyxy_xyxy(roi_noisy_boxes, target_expanded)
    roi_noisy_boxes = torch.cat([*roi_noisy_boxes], dim=0)
    roi_iou = box_iou(roi_noisy_boxes, target_expanded)
    hist_dict = make_histogram(roi_iou)
    iou_dist_dict = add_dict(hist_dict, iou_dist_dict)
    roi_iou = roi_iou.mean().item()

    for stage in range(num_stages):
        if stage == 0 and model_type == "single":
            decode_pred = decode_pred_bbox_xyxy_xyxy(roi_noisy_boxes, pred, (img_size, img_size)).to(device)
            stage_loss, stage_iou = IoUL1_loss(decode_pred, target_expanded.to(device))
        else:
            decode_pred = pred["preds"][stage]
            encode_pred = pred["param_preds"][stage]
            stage_loss, stage_iou = IoU_encoded_L1_loss(
                decode_pred, target_expanded.to(device), encode_pred, target_encoded
            )
        if stage == 0:
            loss = stage_loss
        else:
            loss += (2**stage) * stage_loss
        if batch_idx == 0:
            mean_stage_iou[stage] += stage_iou.item()
            mean_stage_loss[stage] += stage_loss.item()
        else:
            mean_stage_iou[stage] = (mean_stage_iou[stage] + stage_iou.item()) / 2
            mean_stage_loss[stage] = (mean_stage_loss[stage] + stage_loss.item()) / 2
    return mean_stage_iou, mean_stage_loss, iou_dist_dict, roi_iou, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-fc",
        type=int,
        default=3,
        help="Number of FC layers after flattening for FasterRCNN.",
    )
    parser.add_argument(
        "--num-conv",
        type=int,
        default=0,
        help="Number of Convolutional layers after ROI Align for FasterRCNN.",
    )
    parser.add_argument(
        "--roi-feature-size",
        type=int,
        default=11,
        help="Feature map dimension of ROI Align layer for FasterRCNN.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for training.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Height/width used for training.",
    )
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default="model/convnext_tiny_22k_224.pth",
        help="Path to .pth file having ConvNext Backbone for FasterRCNN.",
    )
    parser.add_argument(
        "--pascal-voc-root",
        type=pathlib.Path,
        help="You can use this instead of 'PASCAL_VOC_ROOT' environment variable.",
    )
    parser.add_argument(
        "--box-error-percentage",
        default=30,
        type=int,
        help="Percentage of relative localization error.",
    )
    parser.add_argument(
        "--max-epochs",
        default=100,
        type=int,
        help="Maximum number of epochs for training.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-5,
        type=float,
        help="Learning rate for training FasterRCNN.",
    )
    parser.add_argument(
        "--model-type",
        default="single",
        type=str,
        choices=["single", "multi", "cascade"],
        help="Defines what head architecture to use for LLRN training.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        type=pathlib.Path,
        help="Folder to store checkpoints and tensorboard files.",
    )
    parser.add_argument(
        "--num-stages",
        default=1,
        type=int,
        help="Number of stages for 'multi' and 'cascade' head models.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force removing previously existing output directory.",
    )
    args = parser.parse_args()

    # check for CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    if args.model_type == "single":
        assert args.num_stages == 1, "Can't use multiple stages for 'single' LLRN model."
        model = FasterRCNN(
            weights=args.weights,
            fc_out_features=4096,
            num_conv=args.num_conv,
            num_fc=args.num_fc,
            roi_feature_size=args.roi_feature_size,
            freeze=True,
        )
    elif args.model_type == "multi":
        model = IterModel(
            weights=args.weights,
            fc_out_features=4096,
            num_conv=args.num_conv,
            num_fc=args.num_fc,
            num_stages=args.num_stages,
            roi_feature_size=args.roi_feature_size,
            freeze=True,
        )
    elif args.model_type == "cascade":
        model = CascadeHead(
            weights=args.weights,
            fc_out_features=4096,
            num_conv=args.num_conv,
            num_fc=args.num_fc,
            num_stages=args.num_stages,
            roi_feature_size=args.roi_feature_size,
            freeze=True,
        )

    # prepare dataset
    if args.pascal_voc_root is not None:
        DATA_DIR = args.pascal_voc_root
    elif "PASCAL_VOC_ROOT" in os.environ:
        DATA_DIR = os.environ["PASCAL_VOC_ROOT"]
    else:
        raise KeyError("Could not find PASCAL_VOC_ROOT directory. Please set via env or arg.")
    if os.path.basename(DATA_DIR) == "VOC2012":
        DATA_DIR = os.path.dirname(os.path.dirname(DATA_DIR))
    elif os.path.basename(DATA_DIR) == "VOCdevkit":
        DATA_DIR = os.path.dirname(DATA_DIR)

    train_voc_loader = VOCLoader(
        rootDir=DATA_DIR,
        target_transform=VOCAnnotationTransform,
        imSize=(args.img_size, args.img_size),
        split="train",
        scale=True,
        falseSamplePercentage=100,
        boxErrorPercentage=args.box_error_percentage,
        random_flip=True,
    )
    val_voc_loader = VOCLoader(
        rootDir=DATA_DIR,
        target_transform=VOCAnnotationTransform,
        imSize=(args.img_size, args.img_size),
        split="val",
        scale=True,
        falseSamplePercentage=100,
        boxErrorPercentage=args.box_error_percentage,
        random_sampler=1,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_voc_loader,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=UBBR_collate_fn,
        num_workers=4,
        prefetch_factor=2,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_voc_loader,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=UBBR_collate_fn,
        num_workers=4,
        prefetch_factor=2,
    )

    # initialize the training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, verbose=True, cooldown=2)
    if args.outputs_dir is None:
        output_directory = os.path.join(
            "outputs",
            f"{args.model_type}{args.num_stages}_num_fc_{args.num_fc}"
            + f"_AdamW_lr_{args.learning_rate}_roi_{args.roi_feature_size}",
        )
    else:
        output_directory = args.outputs_dir
    if os.path.isdir(output_directory):
        if args.force:
            shutil.rmtree(output_directory)
        else:
            print(f"ERROR: trying to override existing output: {output_directory}. " "Use --force if that is intended.")
    else:
        os.makedirs(output_directory)

    writer = SummaryWriter(log_dir=output_directory)

    model = model.to(device)

    # start the train loop
    for epoch in range(1, args.max_epochs):
        # parameters for measuring perfomance
        train_bar = tqdm.tqdm(train_data_loader)
        model.train()

        (
            iou_dist_dict,
            mean_stage_iou,
            mean_stage_loss,
            epoch_loss,
        ) = get_empty_epoch_entries(args.num_stages)

        for batch_idx, data in enumerate(train_bar):
            optimizer.zero_grad()
            mean_stage_iou, mean_stage_loss, iou_dist_dict, roi_iou, loss = forward(
                data,
                device,
                model,
                args.num_stages,
                args.model_type,
                args.img_size,
                mean_stage_iou,
                mean_stage_loss,
                batch_idx,
                iou_dist_dict,
            )
            loss.backward()
            optimizer.step()

            epoch_loss = update_progress_bar(
                train_bar,
                "Training",
                mean_stage_iou,
                mean_stage_loss,
                args.num_stages,
                batch_idx,
                loss,
                epoch_loss,
                epoch,
                roi_iou,
            )

        writer.add_scalar("Loss/train total", epoch_loss, epoch)
        for stage in range(args.num_stages):
            writer.add_scalar(f"Loss/train stage {stage}", mean_stage_loss[stage], epoch)
            writer.add_scalar(f"IoU/train stage {stage}", mean_stage_iou[stage], epoch)

        print_dict(iou_dist_dict)

        model.eval()

        (
            iou_dist_dict,
            mean_stage_iou,
            mean_stage_loss,
            epoch_loss,
        ) = get_empty_epoch_entries(args.num_stages)
        val_bar = tqdm.tqdm(val_data_loader)
        for batch_idx, data in enumerate(val_bar):
            with torch.no_grad():  # do not backpropogate for validation epochs
                mean_stage_iou, mean_stage_loss, iou_dist_dict, roi_iou, loss = forward(
                    data,
                    device,
                    model,
                    args.num_stages,
                    args.model_type,
                    args.img_size,
                    mean_stage_iou,
                    mean_stage_loss,
                    batch_idx,
                    iou_dist_dict,
                )

                epoch_loss = update_progress_bar(
                    val_bar,
                    "Validation",
                    mean_stage_iou,
                    mean_stage_loss,
                    args.num_stages,
                    batch_idx,
                    loss,
                    epoch_loss,
                    epoch,
                    roi_iou,
                )

        print_dict(iou_dist_dict)

        scheduler.step(epoch_loss)
        writer.add_scalar("Loss/val total", epoch_loss, epoch)
        for stage in range(args.num_stages):
            writer.add_scalar(f"Loss/val stage {stage}", mean_stage_loss[stage], epoch)
            writer.add_scalar(f"IoU/val stage {stage}", mean_stage_iou[stage], epoch)
        torch.save(
            model.state_dict(),
            output_directory + f"/{epoch}_val_loss_{epoch_loss:0.4f}.pth",
        )

    train_bar.close()
    val_bar.close()
    writer.close()


if __name__ == "__main__":
    main()
