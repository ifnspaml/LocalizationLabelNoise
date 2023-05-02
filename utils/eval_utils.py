from typing import List

import numpy as np
import torch
from terminaltables import AsciiTable
from torchvision.ops import box_area


class IoUEvalClass:
    def __init__(self, classes: list, iter, device="cpu") -> None:
        """Evaluates the IoU seperately for nested and non-nested objects of an inference image.

        Args:
            classes (list): Number of object classes in the dataset, i.e. excluding background
            iter (_type_): number of iteration in the prediction-Dictionary
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        self.iter = iter
        num_classes = len(classes)
        self.classes = classes
        self.device = device
        self.count_instance = torch.zeros(num_classes, dtype=torch.int, device=device)
        self.count_overlap_instances = torch.zeros(num_classes, device=device, dtype=torch.int)
        self.count_non_overlap_instances = torch.zeros(num_classes, device=device, dtype=torch.int)
        self.IoU_all = torch.zeros(size=(iter, num_classes), device=device, dtype=torch.float32)
        self.IoU_overlap = torch.zeros(size=(iter, num_classes), device=device, dtype=torch.float32)
        self.IoU_non_overlap = torch.zeros(size=(iter, num_classes), device=device, dtype=torch.float32)
        self.mean_all = {}
        self.mean_overlap = {}
        self.mean_non_overlap = {}

    def to(self, device) -> None:
        """changes all the attributes to device specified

        Args:
            device (_type_): cuda or cpu
        """
        if device != self.device:
            self.count_instance.to(device)
            self.count_non_overlap_instances.to(device)
            self.count_overlap_instances.to(device)
            self.IoU_all.to(device)
            self.IoU_non_overlap.to(device)
            self.IoU_overlap.to(device)

    def update(
        self,
        iou: List[torch.Tensor],
        iter: int,
        overlap_mask: List[torch.Tensor],
        class_label: List[torch.Tensor],
    ) -> None:
        """updates the class attributes with the given batch of predictions

        Args:
            iou (List[torch.Tensor]): _description_
            iter (int): _description_
            overlap_mask (List[torch.Tensor]): mask of 1D row vector, contains 1 for samples which have nested objects.
            class_label (List[torch.Tensor]): Class label of the sample
        """

        assert iter < self.iter, print(f"iter: {iter} > max_iter: {self.iter}")
        # ensure class_label does not contain bg class label

        unique_label = torch.unique(class_label)
        for label in unique_label:
            label_mask = class_label == label
            overlap = label_mask * overlap_mask
            non_overlap = label_mask * ~overlap_mask
            if iter == 0:
                self.count_instance[label] += torch.count_nonzero(label_mask)
                self.count_overlap_instances[label] += torch.count_nonzero(overlap)
                self.count_non_overlap_instances[label] += torch.count_nonzero(non_overlap)
            self.IoU_all[iter, label] += torch.sum(label_mask * iou)
            self.IoU_overlap[iter, label] += torch.sum(overlap * iou)
            self.IoU_non_overlap[iter, label] += torch.sum(non_overlap * iou)

    def get_mean(self):
        for iter in range(self.iter):
            self.mean_all[iter] = torch.round(self.IoU_all[iter, :] / self.count_instance, decimals=2)
            self.mean_overlap[iter] = torch.round(self.IoU_overlap[iter, :] / self.count_overlap_instances, decimals=2)
            self.mean_non_overlap[iter] = torch.round(
                self.IoU_non_overlap[iter, :] / self.count_non_overlap_instances,
                decimals=2,
            )

    def pretty_print(self, false_iou: List[torch.Tensor]):
        """_summary_

        Args:
            false_iou (List[torch.Tensor]): False IoU should be calculated before calling.
            i.e, false_iou € [0, 1]
        """
        self.to("cpu")
        self.get_mean()
        header = [
            "class",
            "num instances",
            "num_nested_instance",
            "num_non_nested",
            "FalseIoU",
        ]
        for iter in range(self.iter):
            header += [
                f"Iter{iter}_all",
                f"Iter{iter}_nested",
                f"Iter{iter}_non_nested",
            ]
        table_data = [header]
        for j, class_name in enumerate(self.classes):
            row_data = [
                class_name,
                self.count_instance[j].item(),
                self.count_overlap_instances[j].item(),
                self.count_non_overlap_instances[j].item(),
                round(false_iou[j].item(), 2),
            ]
            for i in range(self.iter):
                row_data += [
                    round(self.mean_all[i][j].item(), 2),
                    round(self.mean_overlap[i][j].item(), 2),
                    round(self.mean_non_overlap[i][j].item(), 2),
                ]
            table_data.append(row_data)
        row_data = [
            "Mean",
            round(np.sum(self.count_instance.cpu().numpy()), 2),
            round(np.sum(self.count_overlap_instances.cpu().numpy()), 2),
            round(np.sum(self.count_non_overlap_instances.cpu().numpy()), 2),
            round(np.mean(false_iou.cpu().numpy()), 2),
        ]
        for i in range(self.iter):
            row_data += [
                round(torch.mean(self.mean_all[i]).item(), 2),
                round(torch.mean(self.mean_overlap[i]).item(), 2),
                round(torch.mean(self.mean_overlap[i]).item(), 2),
            ]
        table_data.append(row_data)
        table = AsciiTable(table_data)
        with open("iou_eval_table.txt", "w") as f:
            f.write(table.table)
        print(table.table)


def box_iou(bb1: torch.Tensor, bb2: torch.Tensor):
    """
    Calculates IoU for square matrices, i.e, bb1 & bb2 should have same size

    Args:
        bb1 (torch.Tensor): Tensor with shape (N, 4) & bb1[:, :2] < bb1[:, 2:]
        bb2 (torch.Tensor): Tensor with shape (N, 4)

    Returns:
        _type_: _description_
    """
    assert bb1.size(0) == bb2.size(0), print("boxes should have same size")
    assert bb1.size(1) == 4, print("Boxes should be of form (xmin. ymin, xmax, ymax")
    assert bb2.size(1) == 4, print("Boxes should be of form (xmin. ymin, xmax, ymax")
    area1 = box_area(bb1)
    area2 = box_area(bb2)
    lt = torch.maximum(bb1[:, :2], bb2[:, :2])
    rb = torch.minimum(bb1[:, 2:], bb2[:, 2:])
    inter = rb - lt
    inter = torch.prod(inter, dim=1)
    inter = torch.maximum(torch.zeros_like(inter), inter)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-5)
    return iou
