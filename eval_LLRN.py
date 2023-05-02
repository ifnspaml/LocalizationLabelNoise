import argparse
import os
import pathlib
import warnings

import torch
import tqdm
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes, make_grid

from loaders.cascade_roi import CascadeHead, FasterRCNN, IterModel
from loaders.dataloader import VOC_OBJECT_CLASS_NAMES, VOCAnnotationTransform, VOCLoader
from utils.eval_utils import IoUEvalClass
from utils.misc_utils import is_nested_obj
from utils.utils import UBBR_unreplicate_pred_boxes, collate_fn, de_normalize_img, decode_pred_bbox_xyxy_xyxy, show

warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "llrn_weights",
        type=pathlib.Path,
        default=None,
        help="Path to .pth file of model you want to run inference on.",
    )
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
        default=64,
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
        required=True,
        type=int,
        help="Percentage of relative localization error.",
    )
    parser.add_argument(
        "--model-type",
        default="single",
        type=str,
        choices=["single", "multi", "cascade"],
        help="Defines what head architecture to use for LLRN training.",
    )
    parser.add_argument(
        "--num-stages",
        default=1,
        type=int,
        help="Number of stages for 'multi' and 'cascade' head models. \
              It is also possible to use multiple stages for 'single' model.",
    )
    parser.add_argument(
        "--visualize-every-kth-batch",
        default=None,
        type=int,
        help="If given, every k iterations, writing out the first element of each batch \
              of predictions (green), noisy ground truth (red) and actual GT target \
              (blue). In case of multiple --num-stages, this visualizes only the last \
              stage. Output is stored to directory of given LLRN model.",
    )
    parser.add_argument(
        "--evaluate-on-training-data",
        action="store_true",
        help="For evaluating LLRN performance, typically use noisy validation data. \
              This can be generated with generate_noisy_dataset.py and is generated \
              into a folder like Valplustrain_box_corruption_<xx>. With this flag, you \
              can also evaluate the performance on training data, e.g. to see how \
              much the model generalizes.",
    )
    args = parser.parse_args()

    # check for CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    if args.model_type == "single":
        if args.num_stages > 1:
            print("Using multiple stages for 'single' LLRN model. Is this intended?")
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
        falseSamplePercentage=0,
        boxErrorPercentage=0,
        random_flip=False,
        random_sampler=0,
        annotation_dir=f"Valplustrain_box_corruption_{args.box_error_percentage}",
    )
    val_voc_loader = VOCLoader(
        rootDir=DATA_DIR,
        target_transform=VOCAnnotationTransform,
        imSize=(args.img_size, args.img_size),
        split="val",
        scale=True,
        falseSamplePercentage=0,
        boxErrorPercentage=0,
        random_sampler=0,
        annotation_dir=f"Valplustrain_box_corruption_{args.box_error_percentage}",
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_voc_loader,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=16,
        prefetch_factor=2,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_voc_loader,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=16,
        prefetch_factor=2,
    )

    if args.visualize_every_kth_batch is not None:
        out_dir = os.path.join(
            os.path.dirname(args.llrn_weights),
            f"visualization_box_corruption_{args.box_error_percentage}",
        )
        os.makedirs(out_dir, exist_ok=True)
        print(f"Writing every {args.visualize_every_kth_batch}th batch to {out_dir}")
    model_state_dict = torch.load(args.llrn_weights, map_location="cpu")
    model.load_state_dict(model_state_dict)
    model.to(device).eval()

    false_iou_per_class = torch.zeros(len(VOC_OBJECT_CLASS_NAMES), device=device, dtype=torch.float32)
    iou_eval = IoUEvalClass(VOC_OBJECT_CLASS_NAMES, args.num_stages, device)

    progress_bar = tqdm.tqdm(train_data_loader) if args.evaluate_on_training_data else tqdm.tqdm(val_data_loader)

    with torch.no_grad():
        for batch_idx, data in enumerate(progress_bar):
            image = data[0].to(device)

            target_dict = data[1]
            target_list = target_dict["boxes"]
            target = [box.to(device) for box in target_list]
            gt_boxes = target_dict["GTBoxes"]
            gt_boxes = [box.to(device) for box in gt_boxes]
            class_label = target_dict["labels"]
            class_label = [box.to(device) for box in class_label]
            nested_mask = is_nested_obj(gt_boxes)

            pred = model(image, target)

            target = torch.concat([*target], dim=0).to(device)
            gt_boxes = torch.concat([*gt_boxes], dim=0).to(device)
            class_label = torch.concat([*class_label], dim=0).to(device) - 1

            sample_false_iou = box_iou(target, gt_boxes)
            sample_false_iou = torch.diagonal(sample_false_iou)

            for class_idx, class_name in enumerate(class_label):
                false_iou_per_class[class_name] += sample_false_iou[class_idx]

            for iteration_idx in range(args.num_stages):
                if args.model_type == "single":
                    if iteration_idx > 0:
                        # re-apply 'single' model
                        pred_coords = UBBR_unreplicate_pred_boxes(pred_coords, target_list)
                        pred = model(image, pred_coords)
                    # for iteration 0, noisy targets are used for decoding
                    # for all others, it's previously predicted coordinates
                    pred_coords = decode_pred_bbox_xyxy_xyxy(
                        pred_coords if iteration_idx > 0 else target,
                        pred,
                        (args.img_size, args.img_size),
                    )
                else:
                    pred_coords = pred["preds"][iteration_idx]
                iou = box_iou(pred_coords, gt_boxes)
                iou = torch.diagonal(iou)
                iou_eval.update(iou, iteration_idx, nested_mask, class_label)

            if args.visualize_every_kth_batch is not None and batch_idx % args.visualize_every_kth_batch == 0:
                np_image = de_normalize_img(image[0].squeeze().cpu())
                # network input (noisy labels) in red
                np_image = draw_bounding_boxes(
                    (np_image * 255).type(torch.uint8).squeeze(),
                    target_list[0],
                    colors=(255, 0, 0),
                    width=3,
                )
                num_objects_in_image = target_list[0].shape[0]
                # true labels in blue
                np_image = draw_bounding_boxes(
                    np_image,
                    gt_boxes[:num_objects_in_image],
                    colors=(0, 255, 255),
                    width=3,
                )
                # network output in green
                np_image = draw_bounding_boxes(
                    np_image,
                    pred_coords[:num_objects_in_image].type(torch.int),
                    colors=(0, 255, 0),
                    width=3,
                )
                grid = make_grid(np_image)
                split = "train" if args.evaluate_on_training_data else "val"
                out_filename = os.path.join(out_dir, f"{split}_image_bs{args.batch_size}_batch_{batch_idx}.png")
                show(grid, out_filename, pil=True)
        progress_bar.close()

    # made similar to print_map_summary() of mmdetection
    false_iou_per_class = torch.round(false_iou_per_class / iou_eval.count_instance, decimals=2).cpu()
    iou_eval.pretty_print(false_iou_per_class)


if __name__ == "__main__":
    main()
