ENCODER_L_MODEL = dict(
        type="MSM3D_Encoder",
        in_channels = 6,
        out_channels = 0,
        drop_path = 0.2,
        att_drop_prob = 0.1,
        att_space_filling_curves = [
            'xyz_z-order', 'yxz_z-order',
            'xyz_hilbert', 'yxz_hilbert'],
        shuffle_sfc = True,
        att_block_type = "transformer++",
        channels = (32, 64, 128, 256, 384),
        resnet_layers = (2, 2, 2, 2, 2), 
        att_layers = (0, 0, 0, 2, 2), 
        num_heads = (4, 8, 16, 32, 48),
        window_sizes = (1024, 1024, 1024, 1024, 1024),
        dec_channels = (384, 256, 128, 96, 64),
        dec_resnet_layers = (2, 2, 2, 2, 2),
        dec_att_layers = (2, 2, 0, 0, 0), 
        dec_num_heads = (48, 32, 16, 8, 4),
        dec_window_sizes = (1024, 1024, 1024, 1024, 1024),
    )

BACKBONE_CFG = dict(
    type="MSM3D_EncDec",
    encoder=ENCODER_L_MODEL,
    decoder=dict(
        type="MSM3D_InterpolationDecoder",
        enc_features = [64, 96, 128, 256, 384],
        block_size_levels = [16, 16, 32, 64, 128],
        used_levels = [True, True, True, True, True],
        interpolation_method = "trilinear"
    ),
    encoder_weight_path="",
    encoder_weight_prefix="module.student.",
    encoder_weight_new_prefix=None,
    decoder_weight_path=None,
    decoder_weight_prefix=None,
    decoder_weight_new_prefix=None,
    freeze_encoder=True
)