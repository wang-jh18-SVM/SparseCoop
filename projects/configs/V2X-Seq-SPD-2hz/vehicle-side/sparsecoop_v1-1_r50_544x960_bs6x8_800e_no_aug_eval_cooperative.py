# ================ base config ===================
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

dataset_prefix = "V2X-Seq-SPD-2hz-for-SparseCoop"  # * different for every dataset
v2x_side = "vehicle-side"
num_cams = 1
train_sample_num = 1434  # * different for every dataset
class_names = [
    "car",
    # "truck",
    # "construction_vehicle",
    # "bus",
    # "trailer",
    # "barrier",
    # "motorcycle",
    "bicycle",
    "pedestrian",
    # "traffic_cone",
]
num_classes = len(class_names)

# 48: 6e-4
# 32: 4e-4
# 16: 2e-4
# 8: 1e-4
lr = 2e-4  # * 0.33
total_batch_size = 48
num_gpus = 8
batch_size_per_gpu = total_batch_size // num_gpus
num_iters_per_epoch = int(train_sample_num // (num_gpus * batch_size_per_gpu))
num_epochs = 800
checkpoint_epoch_interval = 40
evaluation_epoch_interval = 40

raw_image_shape = (1920, 1080)
input_shape = (960, 544)

checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        # dict(type="TensorboardLoggerHook"),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='sparsecoop',
        #         name=f'v1-1_{dataset_prefix}_{v2x_side}_lr{lr}_bs{batch_size_per_gpu}x{num_gpus}_{num_epochs}e',
        #     ),
        # ),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)

tracking_val = True
tracking_threshold = 0.2

# ================== model ========================
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
num_anchor = 360
temporal = True
num_temp_instances = int(num_anchor * 2 / 3)
num_output = int(num_anchor / 3)
decouple_attn = True
with_quality_estimation = True

model = dict(
    type="SparseCoop",
    use_grid_mask=False,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint="ckpt/resnet50-19c8e357.pth"),
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    # depth_branch=dict(  # for auxiliary supervision only
    #     type="DenseDepthNet",
    #     embed_dims=embed_dims,
    #     num_depth_layers=num_depth_layers,
    #     loss_weight=0.2,
    # ),
    head=dict(
        type="SparseCoopHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=num_anchor,
            embed_dims=embed_dims,
            anchor=f"data/infos/{dataset_prefix}/{v2x_side}/spd_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=num_temp_instances if temporal else -1,
            default_time_interval=0.5,
            confidence_decay=0.6,
            anchor_grad=False,
            feat_grad=False,
            v2x_side=v2x_side,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=(
            dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            # Deprecated: Only use x and y velocity
            # dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            # Use x,y,z velocity
            dn_noise_scale=[2.0] * 3 + [0.5] * 8,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            # Deprecated: Only use x and y velocity
            # reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            # Use x,y,z velocity
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 5,
            # cls_wise_reg_weights={
            #     class_names.index("traffic_cone"): [
            #         2.0,
            #         2.0,
            #         2.0,
            #         1.0,
            #         1.0,
            #         1.0,
            #         0.0,
            #         0.0,
            #         1.0,
            #         1.0,
            #     ]
            # },
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="GaussianFocalLoss"),
            # cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder", num_output=num_output),
        # Deprecated: Only use x and y velocity
        # reg_weights=[2.0] * 3 + [1.0] * 7,
        # Use x,y,z velocity
        reg_weights=[2.0] * 3 + [1.0] * 8,
    ),
)

# ================== data ========================

file_client_args = dict(backend="disk")

dataset_type = "V2XSeqSPDDataset"
data_root = f"datasets/{dataset_prefix}/cooperative/"
info_root = f"data/infos/{dataset_prefix}/cooperative/"
ann_file_train = info_root + "spd_infos_temporal_train.pkl"
ann_file_val = info_root + "spd_infos_temporal_val.pkl"
splits_data_file = f"data/split_datas/cooperative-split-data-spd.json"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(
    #     type="LoadPointsFromFile",
    #     coord_type="LIDAR",
    #     load_dim=4,
    #     use_dim=3,
    #     file_client_args=file_client_args,
    # ),
    dict(type="ResizeCropFlipImage"),
    # dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(class_names)),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            # "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=[
            "T_global",
            "T_global_inv",
            "timestamp",
            "instance_id",
            "lidar2img",
            # Only for debug
            "sample_idx",
        ],
    ),
]
val_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=["img", "timestamp", "projection_mat", "image_wh"],
        meta_keys=[
            "T_global",
            "T_global_inv",
            "timestamp",
            "lidar2img",
            # Only for debug
            "sample_idx",
        ],
    ),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
    v2x_side=v2x_side,
    splits_data_file=splits_data_file,
    eval_mod=['det', 'track'],
)

data_aug_conf = {
    # "resize_lim": (0.48, 0.54),
    "resize_lim": (0.50, 0.51),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    # "rot_lim": (-5.4, 5.4),
    "rot_lim": (0, 0),
    "H": raw_image_shape[1],
    "W": raw_image_shape[0],
    # "rand_flip": True,
    "rand_flip": False,
    # "rot3d_range": [-0.3925, 0.3925],
    "rot3d_range": (0, 0),
}

data = dict(
    samples_per_gpu=batch_size_per_gpu,
    workers_per_gpu=batch_size_per_gpu * 4,
    train=dict(
        **data_basic_config,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_val,
        tracking_threshold=tracking_threshold,
    ),
    test=dict(
        **data_basic_config,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_val,
        tracking_threshold=tracking_threshold,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)

# ================== eval ========================
vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="Collect", keys=["img"], meta_keys=["timestamp", "lidar2img"]),
]
evaluation = dict(
    interval=num_iters_per_epoch * evaluation_epoch_interval,
    pipeline=vis_pipeline,
    # out_dir="./vis",  # for visualization
)
