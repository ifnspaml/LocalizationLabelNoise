import argparse
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import tqdm
import xmltodict


def make_random_false_box(box: list, boxErrorPercentage, imSize: tuple):
    """generates false boxes by pertubring top, bottom, left, right direction of
    bounding box

    Args:
        box (list): box with [xmin, ymin, xmax, ymax]
        boxErrorPercentage (_type_): Error percentage that needs to be added
        imSize (_type_): (height, width) of the image

    Returns:
        _type_: _description_
    """
    if boxErrorPercentage == 0:
        return box
    new_box = []
    width = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    factor = boxErrorPercentage / 200  # divide by 2 for equal elongation/shrinking on each half
    height_factor = int(height * factor)
    width_factor = int(width * factor)
    if height_factor == 0:
        height_factor = 1
    if width_factor == 0:
        width_factor = 1
    new_box.append(box[0] + np.random.randint(-width_factor, width_factor))
    new_box.append(box[1] + np.random.randint(-height_factor, height_factor))

    new_box[0] = np.clip(new_box[0], 0, box[2] - 1)  # clip min to zero and max to xmax, ymax
    new_box[1] = np.clip(new_box[1], 0, box[3] - 1)

    new_box.append(box[2] + np.random.randint(-width_factor, width_factor))
    new_box.append(box[3] + np.random.randint(-height_factor, height_factor))

    new_box[2] = np.clip(new_box[2], new_box[0], imSize[1] - 1)
    new_box[3] = np.clip(new_box[3], new_box[1], imSize[0] - 1)
    return new_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pascal-voc-root",
        type=pathlib.Path,
        help="You can use this instead of 'PASCAL_VOC_ROOT' environment variable.",
    )
    parser.add_argument(
        "box_error_percentage",
        type=int,
        help="Percentage of relative localization error.",
    )
    parser.add_argument(
        "--generate-noisy-validation-data",
        action="store_true",
        help="For evaluating LLRN performance, we also need noisy validation data. \
              With this option, such data can be generated. It will end up in a \
              folder like Valplustrain_box_corruption_<xx>. This is a different folder \
              than what is used for training LLRNs.",
    )
    args = parser.parse_args()

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

    # path to true annotations
    pascal_base_dir = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
    annot_folder = os.path.join(pascal_base_dir, "Annotations")
    # path mentioning the split - train / val
    split_folder = os.path.join(pascal_base_dir, "ImageSets", "Main")

    # read the names of all file in the annotation folder
    annot_files = os.listdir(annot_folder)
    assert len(annot_files) > 0, f"Could not find annotations in {annot_folder}"
    with open(os.path.join(split_folder, "train.txt"), "r", encoding="utf-8") as file:
        train = file.read()
    with open(os.path.join(split_folder, "val.txt"), "r", encoding="utf-8") as file:
        val = file.read()
    # remove trailing newline, split into list and append xml file extension
    train = [fn + ".xml" for fn in train.rstrip().split("\n")]
    val = [fn + ".xml" for fn in val.rstrip().split("\n")]
    assert len(train) > 0, f"Could not find annotations in {split_folder}/train.txt"
    assert len(val) > 0, f"Could not find annotations in {split_folder}/val.txt"

    # prepare destination folder
    out_dir_str = "Valplustrain" if args.generate_noisy_validation_data else "Annotations"
    dest_folder = os.path.join(
        DATA_DIR,
        "VOCdevkit",
        "VOC2012",
        f"{out_dir_str}_box_corruption_{args.box_error_percentage}",
    )
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)
    # "initialize" with clean annotations, i.e. copy paste to new destination
    for files in annot_files:
        shutil.copy(annot_folder + "/" + files, dest_folder + "/" + files)

    # then, apply random noise and modify files in new destination
    modified_files = train + val if args.generate_noisy_validation_data else train
    print(
        f"Modifying {len(modified_files)} files"
        + (" (including val).." if args.generate_noisy_validation_data else "..")
    )
    for file in tqdm.tqdm(modified_files):
        with open(annot_folder + "/" + file, "r", encoding="utf-8") as xml_file:
            xml_dict = xmltodict.parse(xml_file.read())
        width = int(xml_dict["annotation"]["size"]["width"])  # image width
        height = int(xml_dict["annotation"]["size"]["height"])  # image height
        tree = ET.parse(annot_folder + "/" + file)
        root = ET.ElementTree(tree)

        # iterate through the objects present in the image
        for member in root.findall("object"):
            # get bbox coordinates of each object
            xmin = int(member.find("bndbox/xmin").text)
            ymin = int(member.find("bndbox/ymin").text)
            xmax = int(member.find("bndbox/xmax").text)
            ymax = int(member.find("bndbox/ymax").text)

            # store data in list
            bbox_coordinates = [xmin, ymin, xmax, ymax]
            # falsify coordinates according to error percentage
            false_boxes = make_random_false_box(bbox_coordinates, args.box_error_percentage, imSize=(height, width))

            # change bounding box values in xml file to noisy ones
            member.find("bndbox/xmin").text = str(false_boxes[0])
            member.find("bndbox/xmax").text = str(false_boxes[2])
            member.find("bndbox/ymin").text = str(false_boxes[1])
            member.find("bndbox/ymax").text = str(false_boxes[3])

        # write the changes to file in the dest_folder
        tree.write(dest_folder + "/" + file, encoding="utf-8")


if __name__ == "__main__":
    main()
