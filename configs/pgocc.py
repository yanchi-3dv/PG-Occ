dataset_type = 'NuSceneOVOcc'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/gts'
gt_depth_root = 'data/nuscenes/nuscenes_depth'
retrieval_eval_root = 'data/nuscenes/retrieval_benchmark'

text_prompt_paths = [
    f'./ckpt/text_proto_embeds_clip.pth',
]

processed_data_conf = dict(
    depth_root = 'data/nuscenes_metric3d',
    text_vision_root='data/nuscenes_featup',
    gt_depth_root='data/nuscenes/nuscenes_depth',
)

render_conf = dict(
    use_ov=True,
    render_h=180,
    render_w=320,
    ov_auxi_past_frame_num=0,
    ov_auxi_future_frame_num=0,
)

loss_weights = dict(
    depth_warping=10.0,
    ov_mse=10.0,
    ov_cos=1.0,
    depth_foundation=0.5,
    depth_gt=0.5,
)

eval_thresholds = dict(
    density_threshold=4e-2,
)

point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
occ_size = [200, 200, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_layers_scale_ = [0, 1, 2]
_num_frames_ = 8
_num_queries_ = [4000, 1000, 1000]
_return_intrinsic_ = True

_metric_ = ['miou']
# _metric_ = ['miou', 'depth', 'rayiou']

model = dict(
    type='PGOcc',
    return_gaussians=False,
    metric=_metric_,
    data_aug=dict(
        img_color_aug=True,                 # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    use_grid_mask=False,
    use_mask_camera=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
    pts_bbox_head=dict(
        type='PGOccHead',
        text_prompt_paths=text_prompt_paths,
        use_pca=True,
        render_conf=render_conf,
        loss_weights=loss_weights,
        eval_thresholds=eval_thresholds,
        training_mode='2d',
        class_names=occ_class_names,
        embed_dims=_dim_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        metric=_metric_,
        transformer=dict(
            type='PGOccTransformer',
            embed_dims=_dim_,
            layers_scale=_layers_scale_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            pc_range=point_cloud_range,
            gaussian_scale_range=[],
            occ_size=occ_size,
            render_conf=render_conf,
            use_anisotropy_encoding=True),
        voxelizer=dict(
            type='GaussianVoxelizer',
            vol_range=[-40-0.4, -40-0.4, -1-0.4, 40-0.4, 40-0.4, 5.4-0.4],
            voxel_size=0.4,
            filter_gaussians=True,
            opacity_thresh=0.6, 
            covariance_thresh=1.5e-2)
    ),
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadFeatureFromFiles', root_path=processed_data_conf['depth_root'], key='depth'),
    dict(type='LoadFeatureFromFiles', root_path=processed_data_conf['text_vision_root'], key='text_vision'),
    # dict(type='LoadFeatureFromFiles', root_path=processed_data_conf['gt_depth_root'], key='gt_depth'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, render_inf=True),
    dict(type='GenerateRenderImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, render_conf=render_conf),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D', keys=['img', 't0_2_x_geo', 'text_vision', 'depth', 'render_gt'], # 'gt_depth' is optional
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2img', 'img_timestamp', 'cam2ego', 'ego2lidar', 'lidar2img', 'render_k', 'ori_k', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadFeatureFromFiles', root_path=processed_data_conf['depth_root'], key='depth'),
    dict(type='LoadFeatureFromFiles', root_path=processed_data_conf['text_vision_root'], key='text_vision'),
    # dict(type='LoadFeatureFromFiles', root_path=processed_data_conf['gt_depth_root'], key='gt_depth'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='GenerateRenderImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True, render_conf=render_conf),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D', keys=['mask_camera', 'img', 'voxel_semantics', 'text_vision', 'depth', 'render_gt'],    # add 'gt_depth' if depth evaluation is needed
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2img', 'img_timestamp', 'cam2ego', 'ego2lidar', 'lidar2img', 'render_k', 'ori_k'))
]

data = dict(
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        metric=_metric_,
        pipeline=train_pipeline,
        modality=input_modality,
        return_intrinsic=_return_intrinsic_,
        render_conf=render_conf,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        metric=_metric_,
        pipeline=test_pipeline,
        modality=input_modality,
        return_intrinsic=_return_intrinsic_,
        render_conf=render_conf,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_ov_sweep.pkl',
        metric=_metric_,
        pipeline=test_pipeline,
        modality=input_modality,
        return_intrinsic=_return_intrinsic_,
        render_conf=render_conf,
        test_mode=True),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=350, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

total_epochs = 8
batch_size = 1

# load pretrained weights
load_from = './ckpt/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=2, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=20, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)

# other flags
debug = False