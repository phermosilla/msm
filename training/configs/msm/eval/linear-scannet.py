_base_ = ["../../_base_/default_runtime.py"]

_import_module_ = "../data_root.py"
_import_module_ = "../msm_model.py"

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_20,
)

project_name = "MSM"
group_name = "SSL-Lin"
exp_name = "MSM-Lin-ScanNet"

batch_size =  8 # bs: total bs in all gpus
num_worker =  12
mix_prob = 0.25
accum_grad_steps = 1
clip_grad_norm = 0.0
empty_cache = False
enable_amp = True
evaluate = True
find_unused_parameters = False

encoder_weight_path=MODEL_WEIGHTS

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="MSM3D_EncDec",
        encoder=ENCODER_L_MODEL,
        decoder=dict(
            type="MSM3D_InterpolationDecoder",
            out_channels = 20,
            enc_features = [64, 96, 128, 256, 384],
            block_size_levels = [16, 16, 32, 64, 128],
            used_levels = [True, True, True, True, True],
            interpolation_method = "trilinear",
            linear_layer = True
        ),
        encoder_weight_path=encoder_weight_path,
        encoder_weight_prefix="module.student.",
        encoder_weight_new_prefix=None,
        decoder_weight_path=None,
        decoder_weight_prefix=None,
        decoder_weight_new_prefix=None,
        freeze_encoder=True
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
optimizer = dict(type="AdamW", lr=0.01, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.01,
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

epoch = 200
eval_epoch = 5

# LR
#epoch = 800
#lr_file = DATA_ROOT_PATH+"/scannet_data/tasks/scenes/1.txt"
#epoch = 400
#lr_file = DATA_ROOT_PATH+"/scannet_data/tasks/scenes/5.txt"
#lr_file = DATA_ROOT_PATH+"/scannet_data/tasks/scenes/10.txt"
#lr_file = DATA_ROOT_PATH+"/scannet_data/tasks/scenes/20.txt"
lr_file = None

# LA 
#la_file = DATA_ROOT_PATH+"/scannet_data/tasks/points/points20"
#la_file = DATA_ROOT_PATH+"/scannet_data/tasks/points/points50"
#la_file = DATA_ROOT_PATH+"/scannet_data/tasks/points/points100"
#la_file = DATA_ROOT_PATH+"/scannet_data/tasks/points/points200"
la_file = None

# dataset settings
dataset_type = "ScanNetDataset"
data_root = DATA_ROOT_PATH+"/scannet_data"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=CLASS_LABELS_20,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        lr_file=lr_file,
        la_file=la_file,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
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
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=False,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "displacement"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
             dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "displacement"),
                feat_keys=("color", "normal")
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
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
                mode="test",
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
                keys=("coord", "color", "normal"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "displacement"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
