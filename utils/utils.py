import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def collate_fn(batch):
    """collate function used in Dataloader not using UBBR approach

    Args:
        batch (_type_): _description_

    Returns:
        _type_: _description_
    """
    images, labels, boxes, fBoxes, GTBoxes = [], [], [], [], []
    for i, data in enumerate(batch):
        num_boxes = data[1]["falseBoxes"].size(0)
        images.append(data[0])
        labels.append(data[1]["labels"])
        fBoxes.append(torch.concat([torch.ones((num_boxes, 1)) * i, data[1]["falseBoxes"]], dim=1))
        boxes.append(data[1]["boxes"])
        GTBoxes.append(data[1]["GTBoxes"])
    falseBoxes = torch.concat([*fBoxes], dim=0)
    targetDict = {"falseBoxes": falseBoxes, "boxes": boxes, "labels": labels, "GTBoxes": GTBoxes}
    images = torch.stack([*images])
    return (images, targetDict)


def UBBR_collate_fn(batch):
    """collate function used in Dataloader for UBBR Approach"""
    images, labels, boxes, fBoxes, GTBoxes, GTClass = [], [], [], [], [], []
    for data in batch:
        images.append(data[0])
        labels.append(data[1]["labels"])
        fBoxes.append(data[1]["falseBoxes"])
        boxes.append(data[1]["boxes"])
        GTBoxes.append(data[1]["GTBoxes"])
        GTClass.append(data[1]["GTClass"])
    targetDict = {"falseBoxes": fBoxes, "boxes": boxes, "labels": labels, "GTBoxes": GTBoxes, "GTClass": GTClass}
    images = torch.stack([*images])
    return (images, targetDict)


def UBBR_create_target_boxes(target_box: list, false_boxes: torch.Tensor):
    """UBBR create 50 or more random boxes for a target box. This functions repeates the same
    target for those 50 boxes as label.
    Args:
        target_box (list): List containing target boxes
        false_boxes (torch.Tensor): tensor of False boxes/ random boxes created by UBBR for which label is to be added

    Returns:
        torch.Tensor: Tensor of target boxes having same shape as that of false_boxes
    """
    assert false_boxes.size(1) == 4, print("False Boxes should be torch tensor of size (N, 4)")
    if isinstance(target_box, list):
        target_box = torch.cat([*target_box], dim=0)
    num_fBox_per_target = false_boxes.size(0) // target_box.size(0)
    target_box = target_box.tile(num_fBox_per_target)
    target_box = target_box.view(-1, 4)
    return target_box


def UBBR_replicate_target_boxes(target_box: list, false_boxes: list):
    """deprecated. had the same functionality as UBBR_create_target_boxes()

    Args:
        target_box (list): _description_
        false_boxes (list): _description_

    Returns:
        _type_: _description_
    """
    target_expand = []
    for objects, target_object in zip(false_boxes, target_box):
        for object, target in zip(objects, target_object):
            target_expand.append(UBBR_create_target_boxes(target.unsqueeze(0), object))
    target_expand = torch.cat([*target_expand], dim=0)
    return target_expand


def UBBR_replicate_label_boxes(labels: list, false_boxes: list):
    """Analog for UBBR_create_target_boxes() in case of class label

    Args:
        labels (list): _description_
        false_boxes (list): _description_

    Returns:
        _type_: _description_
    """
    assert isinstance(labels, list), print("Expected a List as label")
    assert isinstance(false_boxes, list), print("Expected list for falseBoxes")
    label_expanded = []
    for label, boxes in zip(
        labels, false_boxes
    ):  # expecting falseBoxes to be of form [[tensor1, tensor2, ..], [tensor1, tensor2, ...], ...]
        for num, box in enumerate(boxes):
            temp_label = label[num].tile(box.size(0))
            label_expanded.append(temp_label)
    label_expanded = torch.cat([*label_expanded], dim=0)
    return label_expanded


def UBBR_unreplicate_pred_boxes(pred_boxes, false_boxes: list):
    """seperates the predicted boxes from the batch according to the number of images.

    Args:
        pred_boxes (_type_): predictions
        false_boxes (list): List of false boxes sorted per image

    Returns:
        _type_: _description_
    """
    assert isinstance(pred_boxes, torch.Tensor), print("pred boxes should be a Tensor")
    assert isinstance(false_boxes, list), print("Expected boxes as instance of list")
    pred_box_list = []
    index = 0
    for boxes in false_boxes:
        if isinstance(boxes, list):
            objects_in_image = []
            for box in boxes:  # this box should be a Tensor
                objects_in_image.append(pred_boxes[index : index + box.size(0), ...])
                index += box.size(0)
            if len(objects_in_image) > 0:
                pred_box_list.append(objects_in_image)
        else:
            pred_box_list.append(pred_boxes[index : index + boxes.size(0), ...])
            index += boxes.size(0)
    return pred_box_list


def encode_xyxy_xyxy(anchors: torch.Tensor, target_boxes: torch.Tensor):
    """Encodes the target / pred boxes according to anchor box
    in the form (xmin, ymin, xmax, ymax)

    Args:
        anchors (torch.Tensor): Anchors for encoding
        targetBoxes (torch.Tensor): Target coordinates which needs to be encoded
    """
    if isinstance(anchors, List):
        anchors = torch.concat([*anchors], dim=0)
    assert anchors.size(-1) == 4, print("Anchor should be (N, 4)")

    if isinstance(target_boxes, List):
        target_boxes = torch.concat([*target_boxes], dim=0)
    assert target_boxes.size(-1) == 4, print("Target Boxes should be (N, 4)")

    encoded_targets = target_boxes.clone()
    anchor_wh = anchors[:, 2:] - anchors[:, :2]
    anchor_wh = anchor_wh.tile(2)
    encoded_targets = encoded_targets - anchors
    encoded_targets = encoded_targets / anchor_wh
    return encoded_targets


def decode_pred_bbox_xyxy_xyxy(anchors: torch.Tensor, preds: torch.Tensor, image_size: tuple):
    """decodes encoded preds according to anchors given. Detaching the pred tensor
    should be done on the calling side
    Args:
        anchors (torch.Tensor): (N, 4) Tensor with entries (x1, y1, x2, y2)
        preds (torch.Tensor): Predicted tensors (N, 4) with entries (t_x1, t_y1, t_x2, t_y2)
        image_size (tuple): tuple containing (H, W) of the input image
    """
    if isinstance(anchors, List):
        anchors = torch.concat([*anchors], dim=0)
    assert anchors.size(-1) == 4, print("Anchor should be (N, 4)")

    anchor_wh = anchors[:, 2:] - anchors[:, :2]
    anchor_wh = anchor_wh.tile(2)

    decoded_pred = preds * anchor_wh + anchors

    clamped_pred = torch.zeros_like(decoded_pred)
    clamped_pred[:, ::2] = torch.clamp(decoded_pred[:, ::2], 0, image_size[1] - 1)
    clamped_pred[:, 1::2] = torch.clamp(decoded_pred[:, 1::2], 0, image_size[0] - 1)

    return clamped_pred


def normalize(boxes: Tensor, mean: List, std: List):
    device = boxes.device
    assert boxes.size(1) == 4, print("boxes should be of size (K, 4)")
    if isinstance(boxes, List):
        boxes = torch.concat([*boxes], dim=0)
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    boxes -= mean.to(device)
    boxes /= std.to(device)
    return boxes


def de_normalize(boxes: Tensor, mean: List, std: List):
    assert boxes.size(1) == 4, print("boxes should be of size (K, 4)")
    device = boxes.device
    if isinstance(boxes, List):
        boxes = torch.concat([*boxes], dim=0)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    boxes *= std
    boxes += mean
    return boxes


def IoU_encoded_L1_loss(boxes1: Tensor, boxes2: Tensor, boxes1_encode: Tensor, boxes2_encode: Tensor):
    assert boxes1.size(1) == 4, print("boxes should be size of (K, 4)")
    assert boxes2.size(1) == 4, print("boxes should be size of (K, 4)")
    assert torch.all(boxes1[:, 1] < boxes1[:, 3]) and torch.all(boxes1[:, 0] < boxes1[:, 2]), print(
        "xmin > xmax impossible"
    )
    assert torch.all(boxes2[:, 1] < boxes2[:, 3]) and torch.all(boxes2[:, 0] < boxes2[:, 2]), print(
        "ymin > ymax Impossible!"
    )
    area_boxes1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
    area_boxes2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])

    intersection = (torch.minimum(boxes1[:, 2], boxes2[:, 2]) - torch.maximum(boxes1[:, 0], boxes2[:, 0])) * (
        torch.minimum(boxes1[:, 3], boxes2[:, 3]) - torch.maximum(boxes1[:, 1], boxes2[:, 1])
    )

    intersection = torch.maximum(torch.zeros_like(intersection), intersection)
    union = area_boxes1 + area_boxes2 - intersection
    min_enclosing_box = (torch.maximum(boxes1[:, 2], boxes2[:, 2]) - torch.minimum(boxes1[:, 0], boxes2[:, 0])) * (
        torch.maximum(boxes1[:, 3], boxes2[:, 3]) - torch.minimum(boxes1[:, 1], boxes2[:, 1])
    )

    iou = intersection / union
    iou_data = iou.data
    giou = iou - ((min_enclosing_box - union) / min_enclosing_box)

    l1 = F.smooth_l1_loss(boxes1_encode, boxes2_encode)
    loss = torch.mean(1 - giou) + l1
    iou_mean = torch.mean(iou_data)
    return loss, iou_mean


def IoUL1_loss(boxes1: Tensor, boxes2: Tensor):
    """implements the classical SmoothL1-GIoU loss

    Args:
        boxes1 (Tensor): predicted bounding box coordinates of shape (N x 4)
        boxes2 (Tensor): GT bounding box coordinates of shape (N x 4)

    Returns:
        _type_: scalar loss value, mean IoU of the boxes
    """
    assert boxes1.size(1) == 4, print("boxes should be size of (K, 4)")
    assert boxes2.size(1) == 4, print("boxes should be size of (K, 4)")
    assert torch.all(boxes1[:, 1] < boxes1[:, 3]) and torch.all(boxes1[:, 0] < boxes1[:, 2]), print(
        "xmin > xmax impossible"
    )
    assert torch.all(boxes2[:, 1] < boxes2[:, 3]) and torch.all(boxes2[:, 0] < boxes2[:, 2]), print(
        "ymin > ymax Impossible!"
    )
    area_boxes1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
    area_boxes2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])

    intersection = (torch.minimum(boxes1[:, 2], boxes2[:, 2]) - torch.maximum(boxes1[:, 0], boxes2[:, 0])) * (
        torch.minimum(boxes1[:, 3], boxes2[:, 3]) - torch.maximum(boxes1[:, 1], boxes2[:, 1])
    )

    intersection = torch.maximum(torch.zeros_like(intersection), intersection)
    union = area_boxes1 + area_boxes2 - intersection
    min_enclosing_box = (torch.maximum(boxes1[:, 2], boxes2[:, 2]) - torch.minimum(boxes1[:, 0], boxes2[:, 0])) * (
        torch.maximum(boxes1[:, 3], boxes2[:, 3]) - torch.minimum(boxes1[:, 1], boxes2[:, 1])
    )

    iou = intersection / union
    iou_data = iou.data
    giou = iou - ((min_enclosing_box - union) / min_enclosing_box)

    norm_boxes1 = boxes1 / 512.0
    norm_boxes2 = boxes2 / 512.0

    l1 = F.smooth_l1_loss(norm_boxes1, norm_boxes2)
    loss = torch.mean(1 - giou) + l1
    iou_mean = torch.mean(iou_data)
    return loss, iou_mean


def de_normalize_img(image: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).type(torch.float32).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).type(torch.float32).unsqueeze(1).unsqueeze(1)
    image *= std
    image += mean
    return image


def normalize_img(image: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).type(torch.float32).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).type(torch.float32).unsqueeze(1).unsqueeze(1)
    image -= mean
    image /= std
    return image


def show(img, filename: str, pil=False):
    check_nested_dir = filename.split("/")
    if len(check_nested_dir) > 1:
        directory = check_nested_dir[0]
        if not os.path.exists(directory):
            os.mkdir(directory)
    if img.requires_grad:
        img = img.detach()
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    if pil:
        Image.fromarray(np.uint8(npimg)).save(filename)
    else:
        plt.imshow(npimg)
        plt.savefig(filename)
