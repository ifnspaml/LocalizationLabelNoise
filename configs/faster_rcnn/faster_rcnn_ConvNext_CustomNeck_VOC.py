_base_ = [
    "../datasets/voc12.py",
    "../../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py",
    "../../mmdetection/configs/_base_/schedules/schedule_1x.py",
    "../../mmdetection/configs/_base_/default_runtime.py",
]

VOC_NUM_CLASSES = 20

# To modify which dataset to use for training, simply update train_dataloader.dataset.ann_subdir
# e.g. via command line --cfg-options train_dataloader.dataset.ann_subdir=Annotations_box_corruption_30
train_dataloader = dict(
    dataset=dict(
        ann_subdir="Annotations",
    ),
)

train_cfg = dict(max_epochs=36)
val_evaluator = dict(
    metric="mAP",
    iou_thrs=[0.5, 0.6, 0.7, 0.8, 0.9],
    eval_mode="area",
)
test_evaluator = val_evaluator

custom_imports = dict(imports=["mmpretrain.models"], allow_failed_imports=False)

model = dict(
    backbone=dict(
        type="mmpretrain.ConvNeXt",
        arch="base",
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth",
        ),
        _delete_=True,
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(bbox_head=dict(num_classes=VOC_NUM_CLASSES)),
)

optim_wrapper = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg={"decay_rate": 0.8, "decay_type": "layer_wise", "num_layers": 12},
    optimizer=dict(
        _delete_=True,
        type="AdamW",
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=36, by_epoch=True, milestones=[27, 33], gamma=0.1),
]
