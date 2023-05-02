# ------------------------------------------------------------------------
# LocalizationLabelNoise
# Copyright (c) 2023 Jonas Uhrig, Jeethesh Pai Umesh. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ssd.pytorch (https://github.com/amdegroot/ssd.pytorch):
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py
# Copyright (c) 2017 Max deGroot, Ellis Brown. All Rights Reserved.
# ------------------------------------------------------------------------

from random import sample
from typing import Callable, Optional

import numpy as np
import torch
import torchvision
from PIL import ImageOps
from torchvision import transforms
from torchvision.datasets import VOCDetection

from utils.eval_utils import box_iou
from utils.utils import UBBR_create_target_boxes

# Dataset split
# Length: Train 5717
# Length: val:  5823
# Length: trainval:  11540

VOC_CLASSES = (
    "bg",  # background at index 0
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# maps VOC class names to their index
VOC_CLASS_TO_INDEX = {class_name: i for i, class_name in enumerate(VOC_CLASSES)}

# names of VOC object classes, i.e. all non-background classes
VOC_OBJECT_CLASS_NAMES = VOC_CLASSES[1:]


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    credits: https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(
        self,
        width: int,
        height: int,
        class_to_ind=None,
        scale=False,
        keep_difficult=False,
    ):
        """Generate a VOCAnnotationTransform object

        Args:
            width (int): _description_
            height (int): _description_
            class_to_ind (_type_, optional): _description_. Defaults to None.
            keep_difficult (bool, optional): _description_. Defaults to False.
            scale (bool, optional): - whether to scale boxes. Dedaults to False
        """
        self.class_to_ind = class_to_ind or VOC_CLASS_TO_INDEX
        self.keep_difficult = keep_difficult
        self.width = width
        self.height = height
        self.scale = scale

    def __call__(self, target: dict):
        """
        Arguments:
            target (dict) : the target annotation to be made usable
                will be an ET.Element
                width (int) : resizing width
                height (int) : resizing height
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        scaleWidth = (self.width - 1) / (int(target["annotation"]["size"]["width"]) - 1)
        scaleHeight = (self.height - 1) / (int(target["annotation"]["size"]["height"]) - 1)
        for obj in target["annotation"]["object"].__iter__():
            difficult = int(obj["difficult"]) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj["name"].lower().strip()
            bbox = obj["bndbox"]

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox[pt])
                # scale height or width
                if self.scale:
                    cur_pt = int(cur_pt * scaleWidth) if i % 2 == 0 else int(cur_pt * scaleHeight)
                    cur_pt = np.clip(cur_pt, 0, self.width - 1) if i % 2 == 0 else np.clip(cur_pt, 0, self.height - 1)
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res


class MyVOCDetection(VOCDetection):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        annotation_dir: str = "Annotations",
    ):
        """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
            Overrides the default VOCDetection class to change the annotation directory
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
            image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
                ``year=="2007"``, can also be ``"test"``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            annotation_dir (str) : Annotation directory name (This directory should be inside default main directory)
        """
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)


class VOCLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        rootDir: str,
        target_transform: VOCAnnotationTransform,
        imSize: tuple,
        split="train",
        scale=False,
        boxErrorPercentage=10,
        falseSamplePercentage=50,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        random_flip=False,
        random_sampler=1,
        annotation_dir=None,
    ) -> None:
        """Generates an object for dataloader stream

        Args:
            rootDir (str): Directory of VOC. Provide the path where "VOCdevkit" is located
            target_transform (VOCAnnotationTransform): Function that converts dict-style objects to lists
            imSize (tuple): tuple(h, w) changes both bounding box as well image size
            split (str): 'train' or 'val'
            scale (bool, optional): Whether to scale image and Boxes
            boxErrorPercentage (0, Optional): [0- 100] Percentage of error in original bbox to be added
            falseSamplePercentage (0, Optional): [0 - 100] Percentage of false samples
            random_sampler: Choose from one of three samplers
                            0 - Random Boxes according to random shift
                            1 - Random Boxes according to UBBR
                            2 - Random boxes by fixing one of the sides or centre
        """
        super().__init__()
        self.rootDir = rootDir
        self.target_transform = target_transform
        self.scale = scale
        if self.scale:
            self.image_transform = torchvision.transforms.Resize(size=(imSize))
        else:
            self.image_transform = None
        self.imSize = (256, 256) if imSize is None else imSize
        self.height, self.width = self.imSize
        self.split = "train" if split == None else split
        self.scale = scale
        self.Gen_Original_GT = VOCDetection(
            root=self.rootDir,
            image_set=self.split,
            download=False,
            transform=self.image_transform,
            target_transform=target_transform(width=self.width, height=self.height, scale=self.scale),
        )
        if annotation_dir is None:
            self.DataGenerator = MyVOCDetection(
                root=self.rootDir,
                image_set=self.split,
                download=False,
                transform=self.image_transform,
                target_transform=target_transform(width=self.width, height=self.height, scale=self.scale),
            )
        else:
            self.DataGenerator = MyVOCDetection(
                root=self.rootDir,
                image_set=self.split,
                download=False,
                transform=self.image_transform,
                target_transform=target_transform(width=self.width, height=self.height, scale=self.scale),
                annotation_dir=annotation_dir,
            )
        # Options for enlarging or compressing the bounding box randomly
        self.fixBoxOptions = ["fixTop", "fixRight", "fixBottom", "fixLeft", "fixCentre"]
        self.mean = mean
        self.std = std
        self.num_boxes = 30
        # percentage of samples which are gonna be false
        self.falseSample = falseSamplePercentage
        self.sampler = 0 if self.falseSample == 0 else 100 // self.falseSample
        self.boxErrorPercentage = boxErrorPercentage
        self.randomFlip = random_flip
        # TODO: this is critical for what kind of input distribution we use
        self.alpha = 0.15  # 0.35 = 50% IoU Boxes, 0.15 = 70% ioU boxes
        self.beta = 0.25  # 0.5 = 50% IoU Boxes, 0.25 = 70% IoU Boxes
        self.random_sampler_options = [
            "make_random_false_box",
            "make_universal_false_boxes",
            "make_simple_false_box",
        ]
        self.random_sampler = random_sampler if random_sampler < 3 else 0
        self.random_sampler_fnc = self.__getattribute__(self.random_sampler_options[random_sampler])
        assert boxErrorPercentage < 100, print("Box percentage = 100 means random box somewhere in the image")

    def box_mirror(self, boxes):
        w = self.width
        flipped_boxes = []
        w_array = np.array([w - 1, 0, w - 1, 0, 0])  # w - xmin, ymin, w - xmax, ymax, label are new coordinates
        for box in boxes:
            box = w_array - np.array(box)
            box = np.abs(box)
            box[:4:2] = np.roll(box[:4:2], 1)
            flipped_boxes.append(box)
        return flipped_boxes

    def __getitem__(self, idx: int) -> tuple:
        """Fetches one tuple containing (Image: H x W x 3, Target: 1 x 5)
        Target is of format [xmin, ymin, xmax, ymax, class], class is integer encoded based on
        VOC_CLASSES variable

        To DO: Introduce different kinds of label noise in later stages

        Args:
            idx (int): index of the data to fetch
        """
        sampleDict = {}
        _, target_org = self.Gen_Original_GT.__getitem__(idx)
        image, target = self.DataGenerator.__getitem__(idx)
        if self.randomFlip:
            flip_counter = np.random.randint(0, 2)
            if flip_counter == 1:
                image = ImageOps.mirror(image)
                target = self.box_mirror(target)
                target_org = self.box_mirror(target_org)
        image = transforms.ToTensor()(image).type(torch.float32)
        image = transforms.Normalize(self.mean, self.std)(image)
        targetTensor = torch.tensor(target)
        sampleDict["boxes"] = torch.tensor(targetTensor[:, :4], dtype=torch.float32)
        sampleDict["labels"] = torch.tensor(targetTensor[:, 4], dtype=torch.int32)
        sampleDict["falseBoxes"] = torch.tensor(targetTensor[:, :4], dtype=torch.float32)
        GTTensor = torch.tensor(target_org)
        sampleDict["GTBoxes"] = torch.tensor(GTTensor[:, :4], dtype=torch.float32)
        sampleDict["GTClass"] = torch.tensor(GTTensor[:, 4], dtype=torch.int32)
        sample

        if self.falseSample == 0 or idx % self.sampler != 0:
            return image, sampleDict
        np.random.seed(idx)
        falseBoxes = [self.random_sampler_fnc(box) for box in target]
        falseBoxes = torch.tensor(falseBoxes).type(torch.float32)
        if self.random_sampler == 1:
            falseBoxes = self.box_sanity_check(falseBoxes, targetTensor, iou_threshold=0.50, min_height=5, min_width=5)
        sampleDict["falseBoxes"] = falseBoxes
        return image, sampleDict

    def box_sanity_check(
        self,
        old_boxes: list,
        target_box: torch.Tensor,
        iou_threshold=0.3,
        min_height=5,
        min_width=5,
    ):
        """Filters out boxes which have IoU threshold less than iou_threshold

        Args:
            old_boxes (list): [xmin, ymin, xmax, ymax] of target box
            target_box (torch.Tensor): 50 possible transformation of above box
            iou_threshold (float, optional): _description_. Defaults to 0.3. choose
            (0.3 for 50% ioU overall boxes, 0.5 for 70% iou overall)
            min_height (int, optional): _description_. Defaults to 5.
            min_width (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        sane_boxes = []
        target = target_box.clone()
        for box, targetBox in zip(old_boxes, target):
            width = box[:, 2] - box[:, 0]
            height = box[:, 3] - box[:, 1]

            # eliminate those boxes whose dimension are less than min_width, min_height
            # this ensures that RoI pooling doesnot encounter any fraction values even at
            # the highest resolution
            width_mask = width >= min_width
            height_mask = height >= min_height
            mask = width_mask * height_mask
            sane_box = box[mask, :]

            # eliminate boxes whose iou is less than iou_threshold
            target_expand = UBBR_create_target_boxes(targetBox[:4].unsqueeze(0), sane_box)
            iou = box_iou(target_expand, sane_box)
            iou_mask = iou > iou_threshold

            # sample ious uniformly
            iou = iou[iou_mask]
            uniform_boxes = self.uniform_sampling(iou, sane_box[iou_mask, :])

            sane_boxes.append(uniform_boxes)
        return sane_boxes

    def uniform_sampling(self, ious, boxes: torch.Tensor):
        """
        Boxes with different IoU are sampled so as to get a uniform
        distribution
        Args:
            ious (_type_): calculated ious of false boxes with target box
            boxes (torch.Tensor): tensors with shape (N, 4) which stores the
                                    coordinates

        Returns:
            _type_: boxes sampled uniformly acc. to IoU with target box
        """
        mask_50_60 = (ious > 0.5) * (ious <= 0.6)
        mask_60_70 = (ious > 0.6) * (ious <= 0.7)
        mask_70_80 = (ious > 0.7) * (ious <= 0.8)
        mask_80_90 = (ious > 0.8) * (ious <= 0.9)
        mask_90_100 = ious > 0.9

        iou_50_60 = ious[mask_50_60]
        iou_60_70 = ious[mask_60_70]
        iou_70_80 = ious[mask_70_80]
        iou_80_90 = ious[mask_80_90]
        iou_90_100 = ious[mask_90_100]

        freq = torch.tensor(
            [
                iou_50_60.size(0),
                iou_60_70.size(0),
                iou_70_80.size(0),
                iou_80_90.size(0),
                iou_90_100.size(0),
            ]
        )
        min_sample = torch.min(freq)
        if min_sample < 2:
            return boxes
        uniform_boxes = torch.cat(
            [
                (boxes[mask_50_60, :])[:min_sample, :],
                (boxes[mask_60_70, :])[:min_sample, :],
                (boxes[mask_70_80, :])[:min_sample, :],
                (boxes[mask_80_90, :])[:min_sample, :],
                (boxes[mask_90_100, :])[:min_sample, :],
            ],
            dim=0,
        )

        return uniform_boxes

    def make_universal_false_boxes(self, box: list):
        """Makes box according to the method mentioned in
        Universal Bounding Box regressor. Generates 50 proposal around the given target box
        https://link.springer.com/chapter/10.1007/978-3-030-20876-9_24

        Args:
            box (list): _description_
        """

        tx = np.random.uniform(-self.alpha, self.alpha, size=(self.num_boxes, 1))
        ty = np.random.uniform(-self.alpha, self.alpha, size=(self.num_boxes, 1))
        ln_beta1 = np.log(1 - self.beta)
        ln_beta2 = np.log(1 + self.beta)
        tw = np.random.uniform(ln_beta1, ln_beta2, size=(self.num_boxes, 1))
        th = np.random.uniform(ln_beta1, ln_beta2, size=(self.num_boxes, 1))

        wg = box[2] - box[0]
        hg = box[3] - box[1]
        xg = int(box[0] + 0.5 * wg)
        yg = int(box[1] + 0.5 * hg)

        xb = np.round(xg + tx * wg)
        yb = np.round(yg + ty * hg)
        wb = np.round(wg * np.exp(tw))
        hb = np.round(hg * np.exp(th))

        xb = np.clip(xb, 0, self.width - 1)
        yb = np.clip(yb, 0, self.height - 1)
        wb = np.clip(wb, 1, self.width - 1)
        hb = np.clip(hb, 1, self.height - 1)

        x1 = np.clip(xb - 0.5 * wb, 0, xb - 1).astype(int)
        x2 = np.clip(xb + 0.5 * wb, xb + 1, self.width - 1).astype(int)
        y1 = np.clip(yb - 0.5 * hb, 0, yb - 1).astype(int)
        y2 = np.clip(yb + 0.5 * hb, yb + 1, self.height - 1).astype(int)

        boxes = np.concatenate([x1, y1, x2, y2], axis=1)
        target_box = np.array(box[:4])[np.newaxis]
        boxes = np.concatenate([boxes, target_box], axis=0)

        return boxes

    def make_random_false_box(self, box: list):
        new_box = []
        width = box[2] - box[0]
        height = box[3] - box[1]
        factor = self.boxErrorPercentage / 200  # divide by 2 for equal elongation/shrinking on each half
        height_factor = int(height * factor)
        width_factor = int(width * factor)
        if height_factor == 0:
            height_factor = 1
        if width_factor == 0:
            width_factor = 1

        new_box.append(box[0] + np.random.randint(-width_factor, width_factor))
        new_box.append(box[1] + np.random.randint(-height_factor, height_factor))

        new_box[0] = np.clip(new_box[0], 0, box[2] - 1)  # clip min to zero and max to xmax, ymax else box collapses
        new_box[1] = np.clip(new_box[1], 0, box[3] - 1)

        new_box.append(box[2] + np.random.randint(-width_factor, width_factor))
        new_box.append(box[3] + np.random.randint(-height_factor, height_factor))

        new_box[2] = np.clip(new_box[2], new_box[0] + 1, self.imSize[1] - 1)
        new_box[3] = np.clip(new_box[3], new_box[1] + 1, self.imSize[0] - 1)
        return new_box

    def make_simple_false_box(self, box: list, option: int) -> list:
        length = box[2] - box[0]
        width = box[3] - box[1]
        dl = int(length * self.boxErrorPercentage / 100) + 1  # avoid going to zero
        dw = int(width * self.boxErrorPercentage / 100) + 1
        rand_dl = np.random.randint(0, dl)  # introduce randomness in the dl and dw
        rand_dw = np.random.randint(0, dw)
        fnc = self.__getattribute__(self.fixBoxOptions[option])
        box = fnc(box, rand_dw, rand_dl)
        return box

    def fixTop(self, box: list, dw, dl):  # ymin or box[1] is fixed
        newBox = [box[0] - int(dw / 2), box[1], box[2] + int(dw / 2), box[3] + dl]
        newBox[0] = max(0, newBox[0])
        newBox[2] = min(newBox[2], self.imSize[1] - 1)  # clip max width
        newBox[3] = min(newBox[3], self.imSize[0] - 1)  # clip max height
        return newBox

    def fixBottom(self, box: list, dw, dl):  # ymax or box[2] is fixed
        newBox = [box[0] - int(dw / 2), box[1] - dl, box[2] + int(dw / 2), box[3]]
        newBox[0] = max(0, newBox[0])
        newBox[1] = max(0, newBox[1])  # clip min height
        newBox[2] = min(newBox[2], self.imSize[1] - 1)  # clip max width
        return newBox

    def fixLeft(self, box: list, dw, dl):  # xmin or box[0] is fixed
        newBox = [box[0], box[1] - int(dl / 2), box[2] + dw, box[3] + int(dl / 2)]
        newBox[1] = max(0, newBox[1])
        newBox[2] = min(newBox[2], self.imSize[1]) - 1  # clip max width
        newBox[3] = min(newBox[3], self.imSize[0] - 1)  # clip max height
        return newBox

    def fixRight(self, box: list, dw, dl):  # xmin or box[2] is fixed
        newBox = [box[0] - dw, box[1] - int(dl / 2), box[2], box[3] + int(dl / 2)]
        newBox[0] = max(0, newBox[0])
        newBox[1] = max(0, newBox[1])
        newBox[3] = min(newBox[3], self.imSize[0] - 1)  # clip max height
        return newBox

    def fixCentre(self, box: list, dw, dl):  # xmin or box[2] is fixed
        newBox = [
            box[0] - int(dw / 2),
            box[1] - int(dl / 2),
            box[2] + int(dw / 2),
            box[3] + int(dl / 2),
        ]
        newBox[0] = max(0, newBox[0])
        newBox[1] = max(0, newBox[1])
        newBox[2] = min(newBox[2], self.imSize[1] - 1)  # clip max width
        newBox[3] = min(newBox[3], self.imSize[0] - 1)  # clip max height
        return newBox

    def normalize(self, image: torch.Tensor):
        image -= self.mean
        image /= self.std
        return image

    def __len__(self):
        return len(self.DataGenerator)
