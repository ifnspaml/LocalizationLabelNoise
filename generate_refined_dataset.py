import argparse
import os
import pathlib
import shutil
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from PIL import Image

from loaders.cascade_roi import CascadeHead, FasterRCNN, IterModel
from utils.utils import decode_pred_bbox_xyxy_xyxy, normalize_img

warnings.simplefilter("ignore")


def scale(bbox, w, h, target_w, target_h):
    scaleHeight = (target_h - 1) / (h - 1)
    scaleWidth = (target_w - 1) / (w - 1)
    new_bbox = []
    for i, _ in enumerate(bbox):
        new_box = int(bbox[i] * scaleWidth) if i % 2 == 0 else int(bbox[i] * scaleHeight)
        new_box = np.clip(new_box, 0, target_w - 1) if i % 2 == 0 else np.clip(new_box, 0, target_h - 1)
        new_bbox.append(new_box)
    return scaleHeight, scaleWidth, new_bbox


def de_scale(bbox, w, h, scale_w, scale_h):
    new_bbox = []
    for i, _ in enumerate(bbox):
        new_box = int(bbox[i] / scale_w) if i % 2 == 0 else int(bbox[i] / scale_h)
        new_box = np.clip(new_box, 0, w - 1) if i % 2 == 0 else np.clip(new_box, 0, h - 1)
        new_bbox.append(new_box)
    return new_bbox


def description_str(description: str):
    """Returns the description string if it does not contain any forbidden characters and
    raises a ValueError otherwise.
    """
    forbidden = [" ", "/", "<", ">", ":", '"', "\\", "|", "?", "*"]
    for char in description:
        if char in forbidden:
            raise ValueError(f"Invalid description {description}: char '{char}' is not allowed")
    return description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "llrn_weights",
        type=pathlib.Path,
        default=None,
        help="Path to .pth file of model you want to run inference on.",
    )
    parser.add_argument(
        "--description",
        required=True,
        type=description_str,
        help="Used for storing refined dataset. Out-dir is build via \
              'Refined_box_corruption_<box_error_percentage>_<model_type><num_stages>_<description>. \
              No empty spaces in description, please.",
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

    model_state_dict = torch.load(args.llrn_weights, map_location="cpu")
    model.load_state_dict(model_state_dict)
    model.to(device).eval()

    pascal_base_dir = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
    image_folder = os.path.join(pascal_base_dir, "JPEGImages")
    annot_folder = os.path.join(pascal_base_dir, "Annotations")
    false_annot_folder = os.path.join(pascal_base_dir, f"Annotations_box_corruption_{args.box_error_percentage}")
    split_folder = os.path.join(pascal_base_dir, "ImageSets", "Main")
    annot_files = os.listdir(annot_folder)
    with open(os.path.join(split_folder, "train.txt"), "r", encoding="utf-8") as file:
        train = file.read()
    # remove trailing newline, split into list and append xml file extension
    train = train.rstrip().split("\n")

    dest_folder = os.path.join(
        pascal_base_dir,
        f"Refined_box_corruption_{args.box_error_percentage}_{args.model_type}{args.num_stages}_{args.description}",
    )
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)
    for file_name in annot_files:
        shutil.copy(os.path.join(annot_folder, file_name), os.path.join(dest_folder, file_name))

    for base_filename in tqdm.tqdm(train):
        image_filename = os.path.join(image_folder, base_filename + ".jpg")
        image = Image.open(image_filename).convert("RGB")
        img_w, img_h = image.size
        image = image.resize((args.img_size, args.img_size))
        tensor_image = F.pil_to_tensor(image).unsqueeze(0).type(torch.float32) / 255.0
        tensor_image = normalize_img(tensor_image).to(device)
        xml_filename = base_filename + ".xml"
        tree = ET.parse(os.path.join(false_annot_folder, xml_filename))
        root = ET.ElementTree(tree)
        for member in root.findall("object"):
            # bbox coordinates
            xmin = int(member.find("bndbox/xmin").text)
            ymin = int(member.find("bndbox/ymin").text)
            xmax = int(member.find("bndbox/xmax").text)
            ymax = int(member.find("bndbox/ymax").text)

            # store data in list
            bbox_coordinates = [xmin, ymin, xmax, ymax]
            scale_h, scale_w, scaled_bbox_list = scale(bbox_coordinates, img_w, img_h, args.img_size, args.img_size)
            scaled_bbox = [torch.tensor(scaled_bbox_list).unsqueeze(0).type(torch.float32).to(device)]
            pred = model(tensor_image, scaled_bbox)

            if args.model_type in ["multi", "cascade"]:
                refined_coords = pred["preds"][args.num_stages - 1].type(torch.int).squeeze().detach().cpu().numpy()
            else:
                target = torch.concat([*scaled_bbox], dim=0)
                decoded_box = decode_pred_bbox_xyxy_xyxy(target, pred, (args.img_size, args.img_size))
                refined_coords = decoded_box.type(torch.int).squeeze().detach().cpu().numpy()
            refined_coords_descaled = de_scale(refined_coords, img_w, img_h, scale_w, scale_h)

            member.find("bndbox/xmin").text = str(refined_coords_descaled[0])
            member.find("bndbox/xmax").text = str(refined_coords_descaled[2])
            member.find("bndbox/ymin").text = str(refined_coords_descaled[1])
            member.find("bndbox/ymax").text = str(refined_coords_descaled[3])
        tree.write(os.path.join(dest_folder, xml_filename), encoding="utf-8")


if __name__ == "__main__":
    main()
