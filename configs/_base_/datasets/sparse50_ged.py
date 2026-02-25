# dataset settings
dataset_type = 'DOTADataset'
# data_root = 'data/split_ss_dota_v15/'
data_root = '/home/flh/projects/seod-main/data/ellipseData/new_ged/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#            'basketball-court', 'storage-tank', 'soccer-ball-field',
#            'roundabout', 'harbor', 'swimming-pool', 'helicopter',
#            'container-crane')
classes = ('ellipse',)
train_pipeline = [
    dict(type='LoadImageFromFile'), 
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= 'data/SparseDet/new_ged_sparse1/sparse50',
        img_prefix='data/ellipseData/new_ged/train_images',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file='data/ellipseData/new_ged/test_annos',
        img_prefix= 'data/ellipseData/new_ged/test_images',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file='data/ellipseData/new_ged/test_annos',
        img_prefix= 'data/ellipseData/new_ged/test_images',
        pipeline=test_pipeline,
        classes=classes))
