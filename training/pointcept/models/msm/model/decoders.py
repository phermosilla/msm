import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import math 

import spconv.pytorch as spconv
from timm.models.layers import trunc_normal_

from pointcept.models.msm.model.layers import SparseNorm, L2NormalizationLayer
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch

@MODELS.register_module("MSM3D_InterpolationDecoder")
class MSM3D_InterpolationDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        enc_features,
        used_levels,
        interpolation_method,
        block_size_levels = None,
        linear_layer = False,
        use_displacement = False,
        conv_decoder = None):

        nn.Module.__init__(self)

        if block_size_levels is None:
            self.block_size_levels = enc_features
        else:
            self.block_size_levels = block_size_levels
        self.linear_layer = linear_layer
        self.used_levels = used_levels

        if linear_layer:
            input_size = 0
            self.norm_layers = nn.ModuleList()
            for cur_iter, cur_layer in enumerate(enc_features):
                if self.used_levels[cur_iter]:
                    input_size += cur_layer
                    self.norm_layers.append(SparseNorm(cur_layer, "layernorm"))

            self.final_stem = nn.Linear(input_size, out_channels)
        self.interpolation_method = interpolation_method

        self.use_displacement = use_displacement

        if not conv_decoder is None:
            self.conv_decoder = build_model(conv_decoder)
        else:
            self.conv_decoder = None

        self.apply(self._init_weights)

    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def interpolate(self, input_coords, displacement, batch_ids, level, feat_grid):

        grid_coords_vox = input_coords.to(torch.float32)/float(level)
        if self.use_displacement:
            grid_coords_vox_offset = (input_coords.to(torch.float32) + displacement + 0.5)/float(level)
        else:
            grid_coords_vox_offset = (input_coords.to(torch.float32) + 0.5)/float(level)
        batch_size = feat_grid.batch_size

        accum_interpolation = []
        for cur_batch in range(batch_size):
            mask_batch = feat_grid.indices[:,0] == cur_batch
            mask_coords_batch = batch_ids == cur_batch

            cur_spatial_shape = (torch.amax(feat_grid.indices[mask_batch,1:], 0)+1).tolist()
            cur_x = spconv.SparseConvTensor(
                features=feat_grid.features[mask_batch],
                indices=torch.cat(
                    [torch.zeros_like(feat_grid.indices[mask_batch,1]).int().reshape((-1,1)), 
                    feat_grid.indices[mask_batch,1:]], dim=1).contiguous(),
                spatial_shape=cur_spatial_shape,
                batch_size=1)

            cur_x = cur_x.dense(False)
            cur_x = torch.permute(cur_x, (0, 4, 1, 2, 3))
            
            div_shape = torch.from_numpy(np.array(cur_x.shape[2:])).to(cur_x.device)
            div_shape = div_shape.reshape((1, 3))
            grid_coords = (grid_coords_vox_offset[mask_coords_batch]/div_shape)*2. - 1.
            
            if self.interpolation_method == "nearest":
                
                cur_coords = grid_coords_vox[mask_coords_batch].to(torch.int32)
                cur_interpolation = cur_x[\
                    torch.zeros_like(batch_ids[mask_coords_batch]), :, \
                    cur_coords[:,0], cur_coords[:,1], cur_coords[:,2]]
                
            elif self.interpolation_method == "trilinear":

                cur_coords = torch.flip(grid_coords.reshape((1, -1, 1, 1, 3)), (-1,))
                cur_interpolation = nn.functional.grid_sample(
                    cur_x, 
                    cur_coords,
                    align_corners=True)
                cur_interpolation = cur_interpolation.reshape((cur_x.shape[1], -1)).transpose(1,0)
            
            accum_interpolation.append(cur_interpolation)

        x = torch.cat(accum_interpolation, 0)

        return x


    def forward(self, x_list, input_dict):

        # Get interpolated features.
        with torch.no_grad():
            input_coords = input_dict["grid_coord"]
            displacement = input_dict["displacement"]
            batch_ids = offset2batch(input_dict["offset"])
            feature_list = []
            for x_iter, x in enumerate(x_list):
                if self.used_levels[x_iter]:
                    level = 2**(x_iter)

                    block_feature_list = []
                    cur_block_size = self.block_size_levels[x_iter]
                    cur_num_blocks = x.features.shape[-1]//cur_block_size
                    bckp_features = x.features.clone()
                    for cur_block in range(cur_num_blocks):
                        x = x.replace_feature(
                            bckp_features[:, cur_block_size*cur_block:cur_block_size*(cur_block+1)])
                        block_feature_list.append(
                            self.interpolate(input_coords, displacement, batch_ids, level, x))
                    x = x.replace_feature(bckp_features)
                    feature_list.append(torch.cat(block_feature_list, -1))
        
        # Normalize.
        if self.linear_layer:
            for cur_iter, cur_feature in enumerate(feature_list):
                feature_list[cur_iter] = self.norm_layers[cur_iter](cur_feature)

        # Concatenate.
        x = torch.cat(feature_list, -1)

        # Return features.
        if self.linear_layer:
            x = self.final_stem(x)
        
        # Conv decoder.
        if not self.conv_decoder is None:
            input_coords = input_dict["grid_coord"]
            batch = offset2batch(input_dict["offset"])
            sparse_shape = torch.add(torch.max(input_coords, dim=0).values, 96).tolist()
            x = spconv.SparseConvTensor(
                features=x,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), input_coords.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1)
            x = self.conv_decoder(x)
        
        return x
    

@MODELS.register_module("MSM3D_PredHead")
class MSM3D_PredHead(nn.Module):
    def __init__(
        self,
        layer_channels = [256, 256, 64],
        layer_activations = ["gelu", None],
        layer_type = ["std", "w_norm"],
        layer_bias = [True, True],
        norm_type = "rmsnorm"):

        nn.Module.__init__(self)

        self.layer_channels = layer_channels
        self.layer_activations = layer_activations
        self.layer_type = layer_type
        self.layer_bias = layer_bias
        
        self.norm_layer = SparseNorm(layer_channels[0], norm_type)

        net_layers = []
        for i in range(len(layer_bias)):
            cur_layer = nn.Linear(
                layer_channels[i], layer_channels[i+1], 
                bias=layer_bias[i])
            if layer_type[i] == "w_norm":
                cur_layer = weight_norm(cur_layer)
                cur_layer.weight_g.data.fill_(1)

            net_layers.append(cur_layer)

            if layer_activations[i] == "l2":
                net_layers.append(L2NormalizationLayer())
            elif layer_activations[i] == "gelu":
                net_layers.append(nn.GELU())

        self.net = nn.Sequential(*net_layers)

        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.norm_layer(x)
        x = self.net(x)
        return x