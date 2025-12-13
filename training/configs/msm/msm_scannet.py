_base_ = ["../_base_/default_runtime.py"]

_import_module_ = "data_root.py"
_import_module_ = "msm_model.py"

project_name = "MSM"
group_name = "SSL"
exp_name = "MSM-ScanNet"

# misc custom setting
epoch = 1800
eval_epoch = 75
batch_size =  12 # bs: total bs in all gpus
num_worker =  24
mix_prob = 0.0
accum_grad_steps = 1
clip_grad_norm = 0.0
empty_cache = False
enable_amp = True
evaluate = True
find_unused_parameters = False

grid_size = 0.02
num_samples = 240000

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SSLSemSegEvaluator", 
            norm_feature_vectors=False,
            decoder=dict(
                type="MSM3D_InterpolationDecoder",
                out_channels = 64+96+128+256+384,
                enc_features = [64, 96, 128, 256, 384],
                block_size_levels = [16, 32, 64, 128, 128],
                used_levels = [True, True, True, True, True],
                interpolation_method = "trilinear",
                linear_layer = False
            )),
    dict(type="CheckpointSaver", save_freq=None),
]

# model settings
ENCODER_L_MODEL['drop_path'] = 0.5
model = dict(
    type="MSMPreTrain",

    backbone=ENCODER_L_MODEL,

    patch_proj_head = [
        dict(type="MSM3D_PredHead",
            layer_channels = [in_feats, in_feats, out_feats],
            layer_activations = ["gelu", None],
            layer_type = ["std", "std"],
            layer_bias = [True, True],
            norm_type = "layernorm")

        for in_feats, out_feats in zip(
            [64, 96, 128, 256, 384], 
            [64, 96, 128, 256, 384])
    ],

    loss = dict(
        type="SSHReconLoss",
        local_weights = [1.0, 1.0, 1.0, 1.0, 1.0]),

    momentum = 0.999,
    wd = 0.05,
    total_iters = 0, #Updated automatically
    mask_ratio = 0.4,
    match_ks = [6, 6, 4, 2, 2], 
    match_radius = [grid_size, grid_size*2, grid_size*4, grid_size*8, grid_size*16],
    project_teacher_patchs = False
)

# scheduler settings
optimizer = dict(type="AdamW", 
                 lr=0.0015, 
                 weight_decay=0.05, 
                 betas=(0.9, 0.95))
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.033,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1.0,
)

# dataset settings
data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type="MultiPreTrainDataset",
        data_folders=[
            DATA_ROOT_PATH+"pretrain_indoor_2cm/scannet_train",
            DATA_ROOT_PATH+"pretrain_indoor_2cm/scannet_val",
            DATA_ROOT_PATH+"pretrain_indoor_2cm/scannet_test",
        ],
        loop=1,
        #clusters_file = DATA_ROOT_PATH+"pretrain_cluster_dict.txt",
        pre_transform = [
            dict(type="CenterShift", apply_z=True),
            dict(type="BoxCrop", box_size=4),
            dict(type="Copy", keys_dict={"coord": "origin_coord"})],
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("origin_coord", "coord", "color", "normal"),
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random", return_full_scene=True),
            dict(type="SphereCrop", point_max=num_samples, mode="random", return_full_scene=True),
            dict(type="NormalizeColor"),
            dict(type="Add", keys_dict={"condition": "Scannet"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("origin_coord", 
                    "coord", 
                    "grid_coord", 
                    "full_origin_coord", 
                    "full_coord", 
                    "full_grid_coord",
                    "condition"),
                offset_keys_dict=dict(
                    offset_crop = "coord", 
                    offset_full= "full_coord"),
                feat_keys=("color", "normal"),
                full_feat_keys=("full_color", "full_normal"),
            ),
        ],
    ),
    val=dict(
        type="ScanNetDataset",
        split="val",
        data_root=DATA_ROOT_PATH+"scannet_data",
        over_segmentation = True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "normal", "segment", "over_seg"),
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=False,
            ),
            dict(type="NormalizeColor"),
            dict(type="Add", keys_dict={"condition": "Scannet"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", 
                      "grid_coord", 
                      "segment", 
                      "over_seg", 
                      "displacement",
                      "condition"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val_2=dict(
        type="ScanNetDataset",
        split="train",
        data_root=DATA_ROOT_PATH+"scannet_data",
        over_segmentation = True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "normal", "segment", "over_seg"),
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=False,
            ),
            dict(type="NormalizeColor"),
            dict(type="Add", keys_dict={"condition": "Scannet"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", 
                      "grid_coord", 
                      "segment", 
                      "over_seg", 
                      "displacement",
                      "condition"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
)

