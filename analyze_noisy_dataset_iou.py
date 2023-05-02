import argparse
import os
import pathlib
from collections import defaultdict
from typing import List, Tuple

import torch
import tqdm
import xmltodict
from terminaltables import AsciiTable
from torchvision.ops import box_iou

from loaders.dataloader import VOC_OBJECT_CLASS_NAMES


def _read_annotation_obj(annotation_obj: dict) -> Tuple[str, List[int]]:
    """Retrieve the (class_name, box) tuple from an annotation object."""
    class_name = annotation_obj["name"]
    xmin = int(annotation_obj["bndbox"]["xmin"])
    ymin = int(annotation_obj["bndbox"]["ymin"])
    xmax = int(annotation_obj["bndbox"]["xmax"])
    ymax = int(annotation_obj["bndbox"]["ymax"])
    return class_name, [xmin, ymin, xmax, ymax]


def _traverse_xml_dict(xml_dict: dict) -> Tuple[List[str], List[List[int]]]:
    class_name_list = []
    bboxes_list = []

    if isinstance(xml_dict["annotation"]["object"], list):
        for ann_object in xml_dict["annotation"]["object"]:
            class_name, box = _read_annotation_obj(ann_object)
            if int(ann_object["difficult"]) == 0:
                class_name_list.append(class_name)
                bboxes_list.append(box)

    else:
        ann_object = xml_dict["annotation"]["object"]
        class_name, box = _read_annotation_obj(ann_object)
        if int(ann_object["difficult"]) == 0:
            class_name_list.append(class_name)
            bboxes_list.append(box)

    return class_name_list, bboxes_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pascal-voc-root",
        type=pathlib.Path,
        help="You can use this instead of 'PASCAL_VOC_ROOT' environment variable.",
    )
    parser.add_argument(
        "--box-error-percentage",
        type=int,
        help="Percentage of relative localization error.",
    )
    parser.add_argument(
        "--noisy-annotation-dir",
        type=pathlib.Path,
        help="Directory containing noisy annotations.",
    )
    args = parser.parse_args()
    if not args.box_error_percentage and not args.noisy_annotation_dir:
        raise KeyError("You need to specify either --noisy-annotation-dir or --box-error-percentage.")

    if args.pascal_voc_root is not None:
        DATA_DIR = args.pascal_voc_root
    elif "PASCAL_VOC_ROOT" in os.environ:
        DATA_DIR = os.environ["PASCAL_VOC_ROOT"]
    else:
        raise KeyError("Could not find PASCAL_VOC_ROOT directory. Please set via env or arg.")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"'{DATA_DIR}' does not refer to an existing directory.")

    annot_folder = os.path.join(DATA_DIR, "Annotations")
    split_folder = os.path.join(DATA_DIR, "ImageSets", "Main")
    annot_files = os.listdir(annot_folder)

    # read the file names of train split
    with open(split_folder + "/train.txt", "r", encoding="utf-8") as file:
        train = file.read()
    train = [fn + ".xml" for fn in train.rstrip().split("\n")]

    noisy_annotation_dir = None
    if args.noisy_annotation_dir:
        if os.path.isdir(args.noisy_annotation_dir):
            noisy_annotation_dir = args.noisy_annotation_dir
        else:
            # args.noisy_annotation_dir may be defined relative to DATA_DIR
            joined_path = os.path.join(DATA_DIR, args.noisy_annotation_dir)
            # check if joining args.noisy_annotation_dir with DATA_DIR yields an existing, absolute path
            if os.path.isdir(joined_path):
                noisy_annotation_dir = os.path.join(DATA_DIR, args.noisy_annotation_dir)
    if noisy_annotation_dir is None and args.box_error_percentage is not None:
        noisy_annotation_dir = os.path.join(DATA_DIR, f"Annotations_box_corruption_{args.box_error_percentage}/")
    noisy_files = os.listdir(noisy_annotation_dir)
    assert len(noisy_files) > 0, "Could not find noisy annotations."

    assert len(noisy_files) == len(annot_files), "Number of GT and noisy annotations do not match"
    true_obj = defaultdict(lambda: [])
    noisy_obj = defaultdict(lambda: [])

    for file in tqdm.tqdm(train):
        temp_true_class_list = []
        temp_true_bboxes_list = []
        temp_noisy_class_list = []
        temp_noisy_bboxes_list = []
        # read the file from the annotation folder
        with open(annot_folder + "/" + file, "r", encoding="utf-8") as xml_file:
            xml_dict = xmltodict.parse(xml_file.read())
        temp_true_class_list, temp_true_bboxes_list = _traverse_xml_dict(xml_dict)

        # read the annotation file from the corrupted folder
        with open(noisy_annotation_dir + "/" + file) as noisy_xml_file:
            noisy_xml_dict = xmltodict.parse(noisy_xml_file.read())
        temp_noisy_class_list, temp_noisy_bboxes_list = _traverse_xml_dict(noisy_xml_dict)

        # check if both class and bbox values are matching
        assert len(temp_true_bboxes_list) == len(temp_true_class_list), "Error extracting GT class and BBox"
        assert len(temp_noisy_bboxes_list) == len(temp_noisy_class_list), "Error extracting noisy class and BBox"

        assert len(temp_true_class_list) == len(temp_noisy_class_list), "Different number of noisy and GT objects."

        for true_class, true_box in zip(temp_true_class_list, temp_true_bboxes_list):
            true_obj[true_class].append(true_box)
        for noisy_class, noisy_box in zip(temp_noisy_class_list, temp_noisy_bboxes_list):
            noisy_obj[noisy_class].append(noisy_box)

    print("Analysing IOU change for detailed description")
    iou_dict = defaultdict(lambda: [])  # maps class names to a list of IoU values
    for voc_class in VOC_OBJECT_CLASS_NAMES:
        gt_bbox = torch.tensor(true_obj[voc_class])
        noisy_bbox = torch.tensor(noisy_obj[voc_class])
        iou_dict[voc_class] = torch.diagonal(box_iou(gt_bbox, noisy_bbox))

    # make similar to print_map_summary() of mmdetection
    header = [
        "class",
        "num instances",
        "IoU",
        "min IoU",
        "Max IoU",
        "Instances > 0.5",
        "Instances > 0.75",
        "Instances < 0.5",
    ]

    iou_table_str_list = [header]
    for voc_class in VOC_OBJECT_CLASS_NAMES:
        row_data = [
            voc_class,
            len(iou_dict[voc_class]),
            round(torch.mean(iou_dict[voc_class]).item(), 2),
            round(torch.amin(iou_dict[voc_class]).item(), 2),
            round(torch.amax(iou_dict[voc_class]).item(), 2),
            torch.count_nonzero(iou_dict[voc_class] >= 0.5).item(),
            torch.count_nonzero(iou_dict[voc_class] >= 0.75).item(),
            torch.count_nonzero(iou_dict[voc_class] < 0.5).item(),
        ]
        iou_table_str_list.append(row_data)
    iou_table_asci = AsciiTable(iou_table_str_list)
    print(iou_table_asci.table)


if __name__ == "__main__":
    main()
