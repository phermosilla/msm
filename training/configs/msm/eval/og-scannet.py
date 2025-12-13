_base_ = ["../../_base_/default_runtime.py"]

_import_module_ = "../data_root.py"
_import_module_ = "../msm_model.py"

project_name = "MSM"
group_name = "SSL-Lin"
exp_name = "MSM-OG-Scannet"

# misc custom setting
epoch = 10
eval_epoch = 10
batch_size =  12 # bs: total bs in all gpus
num_worker =  24
mix_prob = 0.0
accum_grad_steps = 1
clip_grad_norm = 0.0
empty_cache = False
enable_amp = True
evaluate = True
find_unused_parameters = False

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="OGEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
test = dict(type="OGTester", verbose=True)

encoder_weight_path=MODEL_WEIGHTS

# model settings
model = dict(
    type="ObjectGrounding",
    backbone=dict(
        type="MSM3D_EncDec",
        encoder=ENCODER_L_MODEL,
        decoder=dict(
            type="MSM3D_InterpolationDecoder",
            out_channels = 64+96+128+256+384,
            enc_features = [64, 96, 128, 256, 384],
            block_size_levels = [16, 16, 32, 64, 128],
            used_levels = [True, True, True, True, True],
            interpolation_method = "trilinear",
            linear_layer = False
        ),
        encoder_weight_path=encoder_weight_path,
        encoder_weight_prefix="module.student.",
        encoder_weight_new_prefix=None,
        decoder_weight_path=None,
        decoder_weight_prefix=None,
        decoder_weight_new_prefix=None,
        freeze_encoder=True
    ),
    out_bb_dim = 64+96+128+256+384,
    in_feat_dim = 128,
    max_bb = 110
)

# scheduler settings
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.00001, betas=(0.9, 0.95))
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.0005,
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "ScanReferDataset"
data_root = DATA_ROOT_PATH+"/scanrefer"
data_root_scannet = DATA_ROOT_PATH+"/scannet_data"

data = dict(
    num_classes=110,
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
        type=dataset_type,
        split="train",
        data_root=data_root,
        data_root_scannet=data_root_scannet,
        max_bb = 110,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "normal", "instance"),
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=False,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "displacement", "instance"),
                feat_keys=("color", "normal"),
            )],
        test_mode=False,
        test_cfg=None,
        loop=1,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        filter="unique",
        data_root=data_root,
        data_root_scannet=data_root_scannet,
        max_bb = 110,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "normal", "instance"),
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=False,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "displacement", "instance"),
                feat_keys=("color", "normal"),
            )],
        test_mode=False,
        test_cfg=None,
        loop=1,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        #filter="multiple",
        data_root=data_root,
        data_root_scannet=data_root_scannet,
        max_bb = 110,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=False,
                keys=("coord", "color", "normal", "instance"),
            ),
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "displacement", "instance"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
