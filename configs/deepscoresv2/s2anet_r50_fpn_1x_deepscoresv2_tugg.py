# model settings
model = dict(
    type='S2ANetDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='S2ANetHead',
        num_classes=136,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        with_orconv=True,
        # Original config from s2anet
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        # Working config form RCNN
        # anchor_ratios=[0.05, 0.3],
        # anchor_strides=[8, 16, 32, 64, 128],
        # anchor_scales=[1.0],

        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.),
                        clip_border=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.),
                        clip_border=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.1),
    max_per_img=2000)
# dataset settings
dataset_type = 'DeepScoresV2Dataset'
data_root = 'data/deep_scores_dense/'
img_norm_cfg = dict(
    mean = [240, 240, 240],
    std = [57, 57, 57],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(512, 512), threshold_rel=0.6, threshold_abs=200.0),
    dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline,
        use_oriented_bboxes=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        use_oriented_bboxes=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_small.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        use_oriented_bboxes=True))
# evaluation = dict(
#     gt_dir='data/dota/test/labelTxt/', # change it to valset for offline validation
#     imagesetfile='data/dota/test/test.txt')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[50, 100])
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbVisualLoggerHook'),
    ])
# wandb settings
wandb_cfg = dict(
    entity="tuggeluk",
    project='wfcos-testing',
    dryrun=False
)



# runtime settings
total_epochs = 120
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
