from typing import List

import torch
from terminaltables import AsciiTable
from torchvision.ops import box_iou


def make_histogram(tensors: torch.tensor) -> dict:
    """Creates histogram based on different IoU thresholds

    Args:
        tensors (torch.tensor): IoU's
    Returns:
        Dict {'less_30': count,
              '30_50': count,
              ...}
    """
    if tensors.is_cuda:
        tensors = tensors.cpu()
    mask_less_30 = tensors < 0.3
    mask_30_40 = (tensors > 0.3) * (tensors <= 0.4)
    mask_40_50 = (tensors > 0.4) * (tensors <= 0.5)
    mask_50_60 = (tensors > 0.5) * (tensors <= 0.6)
    mask_60_70 = (tensors > 0.6) * (tensors <= 0.7)
    mask_70_80 = (tensors > 0.7) * (tensors <= 0.8)
    mask_80_90 = (tensors > 0.8) * (tensors <= 0.9)
    mask_greater_90 = tensors > 0.9
    return {
        "less_30": torch.count_nonzero(mask_less_30).item(),
        "30_40": torch.count_nonzero(mask_30_40).item(),
        "40_50": torch.count_nonzero(mask_40_50).item(),
        "50_60": torch.count_nonzero(mask_50_60).item(),
        "60_70": torch.count_nonzero(mask_60_70).item(),
        "70_80": torch.count_nonzero(mask_70_80).item(),
        "80_90": torch.count_nonzero(mask_80_90).item(),
        "greater_90": torch.count_nonzero(mask_greater_90).item(),
    }


def add_dict(dict1: dict, dict2: dict) -> dict:
    assert dict1.keys() == dict2.keys(), print(
        f"Dictionary keys dont match, key1: {dict1.keys()}, \
        key2: {dict2.keys()}"
    )
    new_dict = {}
    for key in dict1.keys():
        new_dict[key] = dict1[key] + dict2[key]
    return new_dict


def print_dict(dict1: dict) -> None:
    header = ["IoU Range", "Box Count"]
    table_data = [header]
    for key, val in dict1.items():
        row_data = [key, val]
        table_data.append(row_data)
    table = AsciiTable(table_data)
    print(table.table)


def is_nested_obj(targets: List[torch.Tensor]) -> torch.Tensor:
    nested_mask = []
    for target in targets:
        iou = box_iou(target, target)
        overlap = iou > 0.001
        ones = torch.diag(torch.ones_like(overlap)[:, 0])
        overlap *= ~ones
        nested_mask.append(torch.sum(overlap, dim=-1).type(torch.bool))
    nested_mask = torch.cat([*nested_mask], dim=0)
    return nested_mask
