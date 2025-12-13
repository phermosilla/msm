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

MODEL_WEIGHTS = "../inference/weights/model.pth"