_base_ = ["../../_base_/default_runtime.py"]

from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_200,
)

_import_module_ = "../data_root.py"
_import_module_ = "../msm_model.py"

project_name = "MSM"
group_name = "SSL-Lin"
exp_name = "MSM-NN-ScanNet200"

# misc custom setting
empty_cache = False
evaluate = True
enable_amp = False

encoder_weight_path=MODEL_WEIGHTS

# model settings
model = dict(
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
)

lr_file = None
la_file = None

# Test KNN 
test = dict(
    type="KNNOSSemSegTester",
    k = 1)

# dataset settings
dataset_type = "ScanNet200Dataset"
data_root = DATA_ROOT_PATH+"/scannet_data"

data = dict(
    num_classes=200,
    ignore_index=-1,
    names=CLASS_LABELS_200,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        lr_file=lr_file,
        la_file=la_file,
        over_segmentation = True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "normal", "segment", "over_seg"),
                return_grid_coord=True,
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "displacement", "over_seg"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        over_segmentation = True,
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
                keys=("coord", "color", "normal", "over_seg"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "displacement", "over_seg"),
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
            ],
        ),
    ),
)
