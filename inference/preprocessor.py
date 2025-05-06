from transform import Compose

TRANSFORM=[
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        keys=("coord", "color", "normal"),
        return_grid_coord=True,
        return_full_grid_coord=True
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "color", "grid_coord", "sample_grid_coord"),
        offset_keys_dict=dict(
            offset = "grid_coord", 
            sample_offset= "sample_grid_coord"),
        feat_keys=("color", "normal")),]


def build_preprocessor():
    return Compose(TRANSFORM)