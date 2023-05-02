import argparse
import os
import pathlib
import warnings
import xml.etree.ElementTree as ET

import tqdm
from PIL import Image, ImageDraw

warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pascal-voc-root",
        type=pathlib.Path,
        help="You can use this instead of 'PASCAL_VOC_ROOT' environment variable.",
    )
    parser.add_argument(
        "--box-error-percentage",
        default=None,
        type=int,
        help="Percentage of relative localization error.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        type=str,
        choices=["single", "multi", "cascade"],
        help="Defines what head architecture to use for LLRN training.",
    )
    parser.add_argument(
        "--num-stages",
        default=None,
        type=int,
        help="Number of stages for 'multi' and 'cascade' head models. \
              It is also possible to use multiple stages for 'single' model.",
    )
    parser.add_argument(
        "--description",
        default=None,
        type=str,
        help="Used for finding refined dataset. Directory is build via \
              'Refined_box_corruption_<box_error_percentage>_<model_type><num_stages>_<description>. \
              No empty spaces in description, please.",
    )
    parser.add_argument(
        "--refined-labels-dir",
        default="",
        type=str,
        help="Used for finding refined dataset if desired to visualize it.",
    )
    parser.add_argument(
        "--visualize-training-data",
        action="store_true",
        help="For evaluating LLRN performance, typically use noisy validation data. \
              This can be generated with generate_noisy_dataset.py and is generated \
              into a folder like Valplustrain_box_corruption_<xx>. With this flag, you \
              can also evaluate the performance on training data, e.g. to see how \
              much the model generalizes.",
    )
    parser.add_argument(
        "--visualize-gt",
        action="store_true",
        help="Do you want to see true annotations (GT) in blue?.",
    )
    parser.add_argument(
        "--visualize-noisy-labels",
        action="store_true",
        help="Do you want to see noisy labels in red?.",
    )
    parser.add_argument(
        "--visualize-refined-labels",
        action="store_true",
        help="Do you want to see refined annotations in green?.",
    )
    args = parser.parse_args()

    if args.refined_labels_dir and args.refined_labels_dir.endswith(os.path.sep):
        args.refined_labels_dir = os.path.dirname(args.refined_labels_dir)

    if args.refined_labels_dir and args.visualize_refined_labels and args.box_error_percentage is None:
        box_error_percentage_str = os.path.basename(args.refined_labels_dir).split("_")[3]
        if box_error_percentage_str.isnumeric():
            args.box_error_percentage = int(box_error_percentage_str)
    assert (
        args.box_error_percentage is not None or not args.visualize_noisy_labels
    ), "Need to specify --box-error-percentage if you want to visualize noisy labels"
    assert (
        args.visualize_gt or args.visualize_noisy_labels or args.visualize_refined_labels
    ) == True, "You need to specify which labels you want to visualize via --visualize-[gt,noisy-labels,refined-labels]"

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

    pascal_base_dir = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
    image_folder = os.path.join(pascal_base_dir, "JPEGImages")
    annot_folder = os.path.join(pascal_base_dir, "Annotations")
    assert os.path.isdir(annot_folder) or not args.visualize_gt, f"Could not find annotation directory {annot_folder}"
    split_folder = os.path.join(pascal_base_dir, "ImageSets", "Main")
    with open(os.path.join(split_folder, "train.txt"), "r", encoding="utf-8") as file:
        train = file.read()
    with open(os.path.join(split_folder, "val.txt"), "r", encoding="utf-8") as file:
        val = file.read()
    # remove trailing newline, split into list and append xml file extension
    train = train.rstrip().split("\n")
    val = val.rstrip().split("\n")
    assert (
        len(train) if args.visualize_training_data else len(val)
    ), f"Could not find respective split file content in {split_folder}"

    noisy_labels_folder = os.path.join(pascal_base_dir, f"Valplustrain_box_corruption_{args.box_error_percentage}")
    if not os.path.isdir(noisy_labels_folder) and args.visualize_training_data:
        noisy_labels_folder = os.path.join(pascal_base_dir, f"Annotations_box_corruption_{args.box_error_percentage}")
    assert (
        os.path.isdir(noisy_labels_folder) or not args.visualize_noisy_labels
    ), f"Could not find noisy labels dir: {noisy_labels_folder}"

    refined_labels_folder = args.refined_labels_dir
    if not os.path.isdir(refined_labels_folder) and args.refined_labels_dir:
        refined_labels_folder = os.path.join(pascal_base_dir, args.refined_labels_dir)
    if (
        not os.path.isdir(refined_labels_folder)
        and args.box_error_percentage
        and args.model_type
        and args.num_stages
        and args.description
    ):
        refined_labels_folder = os.path.join(
            pascal_base_dir,
            f"Refined_box_corruption_{args.box_error_percentage}_{args.model_type}{args.num_stages}_{args.description}",
        )

    assert not args.visualize_refined_labels or os.path.isdir(
        refined_labels_folder
    ), "Could not find refined labels dir! Provide via --refined-labels-dir or via individual attributes."

    split = "train" if args.visualize_training_data else "val"
    if args.visualize_refined_labels:
        out_dir = refined_labels_folder + f"_visualization_{split}"
        if args.visualize_gt:
            out_dir += "_gt"
        if args.visualize_noisy_labels:
            out_dir += "_noisy"
    elif args.visualize_noisy_labels:
        out_dir = noisy_labels_folder + f"_visualization_{split}"
        if args.visualize_gt:
            out_dir += "_gt"
    else:
        out_dir = annot_folder + f"_visualization_{split}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing output to {out_dir}")

    progress_bar = tqdm.tqdm(train) if args.visualize_training_data else tqdm.tqdm(val)
    for base_filename in progress_bar:
        image_filename = os.path.join(image_folder, base_filename + ".jpg")
        image = Image.open(image_filename).convert("RGB")
        xml_filename = base_filename + ".xml"

        xml_trees = []
        if args.visualize_gt:
            xml_trees.append(ET.ElementTree(ET.parse(os.path.join(annot_folder, xml_filename))))
        if args.visualize_noisy_labels:
            xml_trees.append(ET.ElementTree(ET.parse(os.path.join(noisy_labels_folder, xml_filename))))
        if args.visualize_refined_labels:
            xml_trees.append(ET.ElementTree(ET.parse(os.path.join(refined_labels_folder, xml_filename))))

        for xml_objects in zip(*[xml_tree.findall("object") for xml_tree in xml_trees]):
            if args.visualize_gt:
                xml_gt_object = xml_objects[0]
            if args.visualize_noisy_labels:
                if args.visualize_gt:
                    xml_noisy_labels = xml_objects[1]
                else:
                    xml_noisy_labels = xml_objects[0]
            if args.visualize_refined_labels:
                if args.visualize_gt and args.visualize_noisy_labels:
                    xml_refined_labels = xml_objects[2]
                elif args.visualize_gt or args.visualize_noisy_labels:
                    xml_refined_labels = xml_objects[1]
                else:
                    xml_refined_labels = xml_objects[0]

            bbox_image = ImageDraw.Draw(image)
            if args.visualize_refined_labels:
                xmin_r = int(xml_refined_labels.find("bndbox/xmin").text)
                ymin_r = int(xml_refined_labels.find("bndbox/ymin").text)
                xmax_r = int(xml_refined_labels.find("bndbox/xmax").text)
                ymax_r = int(xml_refined_labels.find("bndbox/ymax").text)
                bbox_coordinates_refined = [xmin_r, ymin_r, xmax_r, ymax_r]
                width = 1 + args.visualize_noisy_labels + args.visualize_gt
                bbox_image.rectangle(bbox_coordinates_refined, outline="green", width=width)
            if args.visualize_noisy_labels:
                xmin_r = int(xml_noisy_labels.find("bndbox/xmin").text)
                ymin_r = int(xml_noisy_labels.find("bndbox/ymin").text)
                xmax_r = int(xml_noisy_labels.find("bndbox/xmax").text)
                ymax_r = int(xml_noisy_labels.find("bndbox/ymax").text)
                bbox_coordinates_noisy = [xmin_r, ymin_r, xmax_r, ymax_r]
                width = 1 + args.visualize_gt
                bbox_image.rectangle(bbox_coordinates_noisy, outline="red", width=width)
            if args.visualize_gt:
                xmin = int(xml_gt_object.find("bndbox/xmin").text)
                ymin = int(xml_gt_object.find("bndbox/ymin").text)
                xmax = int(xml_gt_object.find("bndbox/xmax").text)
                ymax = int(xml_gt_object.find("bndbox/ymax").text)
                bbox_coordinates_gt = [xmin, ymin, xmax, ymax]
                bbox_image.rectangle(bbox_coordinates_gt, outline="blue")

            image.save(os.path.join(out_dir, f"{base_filename}.jpg"))


if __name__ == "__main__":
    main()
