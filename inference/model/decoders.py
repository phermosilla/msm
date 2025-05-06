import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import math 

import spconv.pytorch as spconv

from .layers import SparseNorm

from .utils import offset2batch, MODELS, build_model

@MODELS.register_module("MSM3D_InterpolationDecoder")
class MSM3D_InterpolationDecoder(nn.Module):
    def __init__(
        self,
        enc_features,
        used_levels,
        interpolation_method,
        block_size_levels = None):

        nn.Module.__init__(self)

        if block_size_levels is None:
            self.block_size_levels = enc_features
        else:
            self.block_size_levels = block_size_levels
        self.used_levels = used_levels

        self.interpolation_method = interpolation_method


    def interpolate(self, input_coords, batch_ids, level, feat_grid):

        grid_coords_vox = input_coords.to(torch.float32)/float(level)
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
                            self.interpolate(input_coords, batch_ids, level, x))
                    x = x.replace_feature(bckp_features)
                    feature_list.append(torch.cat(block_feature_list, -1))

        # Concatenate.
        x = torch.cat(feature_list, -1)
        
        return x