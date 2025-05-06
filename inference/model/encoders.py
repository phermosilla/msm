import torch
import torch.nn as nn
import numpy as np
from functools import partial

import spconv.pytorch as spconv

from .layers import SparseNorm
from .layers import SpaceFillingCurves, SpaceFillingCurves2Sparse
from .layers import ResFormer, ResNet, SparseSkipConnection

from .utils import offset2batch, MODELS


@MODELS.register_module("MSM3D_Encoder")
class MSM3D_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop_path = 0.4,
        att_drop_prob = 0.0,
        att_space_filling_curves = [
            'xyz_z-order', 'yxz_z-order',
            'xyz_hilbert', 'yxz_hilbert'],
        shuffle_sfc = True,
        att_block_type = "transformer++",
        channels = (32, 64, 128, 256, 384),
        resnet_layers = (2, 2, 2, 2, 2),
        att_layers = (2, 2, 2, 2, 2),
        num_heads = (2, 4, 8, 16, 32),
        window_sizes = (1024, 1024, 1024, 1024, 1024),
        dec_channels = (384, 256, 128, 96, 96),
        dec_resnet_layers = (2, 2, 2, 2, 2),
        dec_att_layers = (2, 2, 2, 2, 2),
        dec_num_heads = (32, 16, 8, 4, 2),
        dec_window_sizes = (1024, 1024, 1024, 1024, 1024)):

        nn.Module.__init__(self)

        # Save parameters.
        self.window_sizes = window_sizes
        self.resnet_layers = resnet_layers
        self.att_layers = att_layers
        self.dec_resnet_layers = dec_resnet_layers
        self.dec_att_layers = dec_att_layers

        # Attention block type.
        if att_block_type == "transformer":
            bias = True
            mlp_scale = 4
            norm_type = "layernorm"
            mlp_type = "standard"
        elif att_block_type == "transformer++":
            bias = False
            mlp_scale = 4
            norm_type = "rmsnorm"
            mlp_type = "glu"
        
        norm_layer = partial(
            SparseNorm, 
            type=norm_type, 
            bias=bias)

        # Create the encoder.
        self.init_transform = torch.nn.Sequential(
            torch.nn.Linear(in_channels, channels[0]),
            norm_layer(channels[0]))
        
        drop_path_iter = 0
        drop_paths_enc = np.linspace(0, drop_path, np.sum(resnet_layers)+np.sum(att_layers))
        prev_feats = channels[0]
        self.num_stages_enc = len(att_layers)
        self.down = nn.ModuleList()
        self.registers_down = nn.ModuleList()
        self.pos_enc = nn.ModuleList()
        self.resnet_enc = nn.ModuleList()
        self.att_enc = nn.ModuleList()
        for s in range(self.num_stages_enc):
            cur_feats = channels[s]

            if s > 0:
                # Downsample.
                self.down.append(
                    spconv.SparseSequential(
                        spconv.SparseConv3d(
                            prev_feats,
                            cur_feats,
                            kernel_size=2,
                            stride=2,
                            bias=True,
                            indice_key=f"spconv{s}"),
                        norm_layer(cur_feats)))
            
            # Compute resnet layers.
            if resnet_layers[s] > 0:
                resnet_list = [
                    ResNet(
                        cur_feats,
                        indice_key = f"smconv{s}",
                        path_drop_prob = drop_paths_enc[drop_path_iter+i],
                        kernel_size = 3,
                        post_norm = True,
                        bias = bias,
                        norm_layer = norm_layer)
                    for i in range(resnet_layers[s])]
                self.resnet_enc.append(nn.Sequential(*resnet_list))
            else:
                self.resnet_enc.append(spconv.Identity())
            drop_path_iter += resnet_layers[s]

            # Compute attention layers.
            if att_layers[s] > 0:
                att_list = [SpaceFillingCurves(att_space_filling_curves, shuffle_sfc)]
                att_list = att_list + [
                    ResFormer(
                        num_features = cur_feats,
                        num_heads = num_heads[s],
                        window_size = window_sizes[s],
                        path_drop_prob = drop_paths_enc[drop_path_iter+i],
                        att_drop_out = att_drop_prob,
                        bias = bias,
                        mlp_scale = mlp_scale,
                        norm_layer = norm_layer,
                        mlp_type = mlp_type)
                    for i in range(att_layers[s])]
                att_list.append(SpaceFillingCurves2Sparse())
                self.att_enc.append(nn.Sequential(*att_list))
            else:
                self.att_enc.append(spconv.Identity())
            
            prev_feats = cur_feats
            drop_path_iter += att_layers[s]


        # Create the decoder.
        self.decoder_init_norm = norm_layer(dec_channels[0])
        drop_path_iter = 0
        drop_paths_dec = np.linspace(0, drop_path, np.sum(dec_resnet_layers)+np.sum(dec_att_layers))
        drop_paths_dec = drop_paths_dec[::-1]
        self.num_stage_dec = len(dec_att_layers)
        self.up = nn.ModuleList()
        self.att_dec = nn.ModuleList()
        self.resnet_dec = nn.ModuleList()
        self.linears_skip = nn.ModuleList()
        self.skip = nn.ModuleList()
        for s in range(self.num_stage_dec):
            cur_feats = dec_channels[s]

            if s > 0:

                # Upsample.
                self.up.append(
                    spconv.SparseSequential(
                        norm_layer(prev_feats),
                        spconv.SparseInverseConv3d(
                            prev_feats,
                            cur_feats,
                            kernel_size=2,
                            bias=True,
                            indice_key=f"spconv{self.num_stages_enc - s}")))
                
                # Skip connection.
                cur_feats_enc = channels[self.num_stages_enc - s -1]
                
                # Linear for skip connections.
                self.linears_skip.append(
                    spconv.SparseSequential(
                        norm_layer(cur_feats_enc),
                        nn.Linear(cur_feats_enc, cur_feats)))
                
                # Skip connection.
                self.skip.append(SparseSkipConnection(0.0, cur_feats, 1.0))

            # Compute resnet layers.
            if dec_resnet_layers[s] > 0:
                resnet_list = [
                    ResNet(
                        cur_feats,
                        indice_key = f"smconv{len(dec_resnet_layers) -s -1}",
                        path_drop_prob = drop_paths_dec[drop_path_iter+i],
                        kernel_size = 3,
                        post_norm = True,
                        bias = bias,
                        norm_layer = norm_layer)
                    for i in range(dec_resnet_layers[s])]
                self.resnet_dec.append(nn.Sequential(*resnet_list))
            else:
                self.resnet_dec.append(spconv.Identity())
            drop_path_iter += dec_resnet_layers[s]

            # Compute attention layers.
            if dec_att_layers[s] > 0:
                att_list = [SpaceFillingCurves(att_space_filling_curves, shuffle_sfc)]
                att_list = att_list + [
                    ResFormer(
                        num_features = cur_feats,
                        num_heads = dec_num_heads[s],
                        window_size = dec_window_sizes[s],
                        path_drop_prob = drop_paths_dec[drop_path_iter+i],
                        att_drop_out = att_drop_prob,
                        bias = bias,
                        mlp_scale = mlp_scale,
                        norm_layer = norm_layer,
                        mlp_type = mlp_type)
                    for i in range(dec_att_layers[s])]
                att_list.append(SpaceFillingCurves2Sparse())
                self.att_dec.append(nn.Sequential(*att_list))
            else:
                self.att_dec.append(spconv.Identity())

            prev_feats = cur_feats
            drop_path_iter += dec_att_layers[s]

        if out_channels > 0:
            self.final_stem = torch.nn.Linear(
                dec_channels[-1], out_channels)
        else:
            self.final_stem = None

    
    def set_condition(self, condition):
        if not self.ppt_context_manager is None:
            self.ppt_context_manager.set_condition(condition)


    def create_sparse_tensor(self, input_dict):

        discrete_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

        if "condition" in input_dict:
            self.set_condition(input_dict["condition"][0])

        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1)
        return x


    def forward_encoder(self, x):

        # Init transform.
        x = x.replace_feature(self.init_transform(x.features))

        # Encoder.
        out_enc_feats = []
        for s in range(self.num_stages_enc):

            # Downscale.
            if s > 0:
                x = self.down[s-1](x)

            # ResNet.
            x = self.resnet_enc[s](x)

            # Attention layers.
            if self.att_layers[s] > 0:
                x = self.att_enc[s](x)
            out_enc_feats.append(x)

        return out_enc_feats
    

    def forward_decoder(self, encoder_features):

        # Decoder
        out_feats = []
        encoder_features.reverse()
        x = self.decoder_init_norm(encoder_features[0])
        for s in range(self.num_stage_dec):

            # Downscale.
            if s > 0:
                x = self.up[s-1](x)
                skip_x = self.linears_skip[s-1](encoder_features[s])
                x = self.skip[s-1](x, skip_x)

            # ResNet.
            x = self.resnet_dec[s](x)

            # Attention layers.
            if self.dec_att_layers[s] > 0:
                x = self.att_dec[s](x)
            out_feats.append(x)

        # Final stem.
        if not self.final_stem is None:
            x = x.replace_feature(self.final_stem(x.features))
            out_feats.append(x)

        out_feats.reverse()

        return x, out_feats


    def forward(self, x):
        
        # Create the input tensor.
        if isinstance(x, dict):
            x = self.create_sparse_tensor(x)

        # Encoder.
        out_enc_feats = self.forward_encoder(x)

        # Decoder.
        x, out_feats = self.forward_decoder(out_enc_feats)
       
        return x, out_feats
