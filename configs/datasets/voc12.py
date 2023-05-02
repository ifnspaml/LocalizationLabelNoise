_base_ = [
    "../../mmdetection/configs/_base_/datasets/voc0712.py",
]

# dataset settings
import os

if "PASCAL_VOC_ROOT" in os.environ:
    # must point to parent of /<path_to>/PascalVOC/VOCdevkit/VOC2012 to work with mmdet dataset
    data_root = os.path.dirname(os.environ["PASCAL_VOC_ROOT"])
else:
    raise KeyError(
        "Could not find PASCAL_VOC_ROOT as environment variable. Please set accordingly."
    )


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        data_root=data_root,
        ann_file=os.path.join(data_root, "VOC2012", "ImageSets", "Main", "train.txt"),
        data_prefix=dict(sub_data_root="VOC2012/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline={{_base_.train_pipeline}},
        backend_args={{_base_.backend_args}},
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type={{_base_.dataset_type}},
        data_root=data_root,
        ann_file=os.path.join(data_root, "VOC2012", "ImageSets", "Main", "val.txt"),
        data_prefix=dict(sub_data_root="VOC2012/"),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        backend_args={{_base_.backend_args}},
    ),
)
test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type="VOCMetric", metric="mAP", eval_mode="11points")
test_evaluator = val_evaluator
