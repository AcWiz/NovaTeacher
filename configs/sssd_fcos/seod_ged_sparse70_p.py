_base_ = "base_ged_default.py"

# data_root = 'data/split_ss_dota_v15/'
data_root = 'data/ellipseData/new_ged/'

classes = ('ellipse',)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        sup=dict(
            ann_file= 'data/SparseDet/new_ged_sparse1/sparse70',
            img_prefix=data_root + 'train_images/',
            classes=classes,
        ),
        unsup=dict(
            ann_file= 'data/SparseDet/new_ged_sparse1/sparse70',
            img_prefix=data_root + 'train_images/',
            classes=classes,
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[4, 4],
        )
    ),
)

model = dict(
    semi_loss=dict(type='RotatedSingleStageDTLoss', loss_type='pr_origin_p5',
                   cls_loss_type='bce', dynamic_weight='50ang',
                   aux_loss='ot_loss_norm', aux_loss_cfg=dict(clamp_ot=True)),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=6400,
    )
)





log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',
             interval=10,
             by_epoch=True),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="ssad_fcos",
        #         name="Default",
        #         config=dict(
        #             work_dirs="${work_dir}",
        #             total_step="${runner.max_iters}",
        #         ),
        #     ),
        #     by_epoch=False,
        # ), 
    ])