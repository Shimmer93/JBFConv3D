_base_ = '../_base_/default_runtime.py'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=19,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=120,
        dropout_ratio=0.5,
        average_clips='prob'))

dataset_type = 'JBFDataset'
ann_file = '/home/zpengac/datasets/har/ntu/ntu60_2d_jbf.pkl'
jbf_dir = '/home/zpengac/datasets/har/ntu/jbf'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='JBFDecode'),
    dict(type='JBFCompactResizePad', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(64, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='JBFFlip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GenerateJBFTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False, 
        with_jmv=True, 
        with_bm=True, 
        with_fm=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='JBFDecode'),
    dict(type='JBFCompactResizePad', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64)),
    # dict(type='CenterCrop', crop_size=64),
    dict(
        type='GenerateJBFTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False, 
        with_jmv=True, 
        with_bm=True, 
        with_fm=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(
    #     type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='JBFDecode'),
    dict(type='JBFCompactResizePad', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64)),
    # dict(type='CenterCrop', crop_size=64),
    dict(
        type='GenerateJBFTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp, 
        with_jmv=True, 
        with_bm=True, 
        with_fm=True),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            jbf_dir=jbf_dir,
            split='xsub_train',
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        jbf_dir=jbf_dir,
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        jbf_dir=jbf_dir,
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=24, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=24,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2))
