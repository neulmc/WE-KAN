_base_ = [
    '../_base_/datasets/challenge_bs8_600.py', '../_base_/schedules/schedule_1x_adamwn.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'
high_ratio = 1.5

# model settings
model = dict(
    type='FCOS_tank',
    high_ratio = high_ratio,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    backbone2=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    neck2=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),

    bbox_head=dict(
        type='RotatedFCOSHead_nbnew_kanfuse_tank',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128, 8/high_ratio, 16/high_ratio, 32/high_ratio, 64/high_ratio, 128/high_ratio,],
        regress_ranges = (
        # backbone1 (标准分辨率) - 主要负责主流尺度
        (-1, 32), (32, 64), (64, 128), (128, 256), (256, 1e6),
        # backbone2 (高分辨率) - 专注小目标和超大目标，与backbone1形成重叠
        (-1, 32/high_ratio), (32/high_ratio, 64/high_ratio), (64/high_ratio, 128/high_ratio), (128/high_ratio, 256/high_ratio), (256/high_ratio, 1e6),),
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        angle_coder=dict(  # lmc code3
            type='UCResolver',
            angle_version=angle_version,
            mdim=3,
            invalid_thr=0.2,
            loss_angle_restrict=dict(
                type='mmdet.L1Loss', loss_weight=0.05), # 改回0.05
                #type='mmdet.L1Loss', loss_weight=0.01),
        ),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_kan=dict(  # lmc KAN new
            #type='KANIdenty', loss_weight=0.05),
             type='KANIdenty', loss_weight=0.0),  # 改为0
        kan=True,  # lmc KAN
        nb_weight=True,  # lmc NB
        nb_pkl='/home/lmc/code/RSAR/bayes_dataset/gmm_naive_bayes_log_features.pkl',    # lmc NB
    ),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        #score_thr=0.05,
        score_thr=0.01,
        nms=dict(type='nms_rotated', iou_threshold=0.5),
        max_per_img=2000))
