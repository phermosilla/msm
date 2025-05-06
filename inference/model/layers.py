import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import spconv.pytorch as spconv
from flash_attn import flash_attn_varlen_qkvpacked_func

from .serialization import encode

epsilon_init = 1e-1
epsilon_reg = 1e-1

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Code: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    

class SparseNorm(spconv.SparseModule):

    def __init__(
        self,
        num_features,
        type="batchnorm",
        bias = True):
        
        super().__init__()

        self.norm = None
        if type=="batchnorm":
            self.norm = nn.BatchNorm1d(num_features, eps=1e-4)
        elif type=="layernorm":
            self.norm = nn.LayerNorm(num_features, eps=1e-4)
        elif type=="rmsnorm":
            self.norm = RMSNorm(num_features, eps=1e-4, bias = bias)

    def forward(self, x):

        if self.norm is None:
            return x
        
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(self.norm(x.features))
        else:
            return self.norm(x)
    
    
class SparseDropPath(spconv.SparseModule):

    def __init__(self, drop_prob):
        
        self.drop_prob = drop_prob
        super().__init__()

    def forward(self, x):

        if self.drop_prob == 0. or not self.training:
            return x
        
        batch_size = x.indices[:, 0].max().long()+1
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + \
            torch.rand((batch_size,), 
                        dtype=x.features.dtype, 
                        device=x.features.device)
        random_tensor.floor_()  # binarize
        random_tensor = torch.index_select(
                random_tensor, 0, x.indices[:, 0].long())
        
        return x.replace_feature(x.features.div(keep_prob) \
                * random_tensor.reshape((-1,1)))
    

class SparseSkipConnection(spconv.SparseModule):

    def __init__(self,  
        drop_prob,
        num_features,
        init_gamma= epsilon_init):

        super().__init__()
        
        self.drop_path = SparseDropPath(drop_prob)
        self.gamma = nn.Parameter(
            init_gamma * torch.ones((1, num_features)))

    def forward(self, x, y):

        x = x.replace_feature(x.features*self.gamma)
        x = x.replace_feature(self.drop_path(x).features + y.features)
        return x
    

class ResNet(spconv.SparseModule):

    def __init__(self, 
        num_features,
        indice_key = None,
        path_drop_prob = 0.0,
        kernel_size = 3,
        post_norm = False,
        bias = True,
        norm_layer = None):

        super().__init__()

        self.conv1 = spconv.SubMConv3d(
            num_features,
            num_features,
            kernel_size=kernel_size,
            bias=bias,
            indice_key=indice_key)
        self.conv2 = spconv.SubMConv3d(
            num_features,
            num_features,
            kernel_size=kernel_size,
            bias=bias,
            indice_key=indice_key)

        self.actfunc = nn.GELU()

        self.norm = norm_layer(num_features)
        self.skip_path = SparseSkipConnection(path_drop_prob, num_features)

        self.post_norm = post_norm
        if self.post_norm:
            self.norm_post_layer = norm_layer(num_features)


    def forward(self, x):
        
        out = x
        out = self.norm(out)
        out = self.conv1(out)
        out = out.replace_feature(self.actfunc(out.features))
        out = self.conv2(out)
        if self.post_norm:
            out = self.norm_post_layer(out)
        return self.skip_path(out, x)
    

class SpaceFillingCurves(nn.Module):

    def __init__(self,
                 space_filling_curves=[],
                 shuffle = False):
        self.space_filling_curves = space_filling_curves
        self.shuffle = shuffle
        super().__init__()

    def forward(self, inputs):
        x = inputs
        
        discrete_coords = x.indices
        codes = []

        # Create the batch ids.
        batch_ids = discrete_coords[:,0]

        # Batch offsets.
        batch_counts = batch_ids.bincount() 
        batch_offsets_no_reg = nn.functional.pad(
            input=torch.cumsum(batch_counts, dim=0), 
            pad=(1, 0), mode='constant', value=0).to(torch.int32)
        batch_offsets = torch.cumsum(batch_counts, dim=0).long()
        batch_offsets = nn.functional.pad(
            input=batch_offsets, pad=(1, 0), 
            mode='constant', value=0).to(torch.int32)
        max_counts = torch.amax(batch_counts).item()

        # Space filling curves.
        sfc_list = np.arange(len(self.space_filling_curves))
        if self.shuffle:
            sfc_list = np.random.permutation(sfc_list)
        for cur_sfc_i in sfc_list:
            cur_sfc = self.space_filling_curves[cur_sfc_i]
            if cur_sfc.startswith('xyz'):
                cur_ids = discrete_coords[:,1:]
            elif cur_sfc.startswith('xzy'):
                cur_ids = discrete_coords[:,[1,3,2]]
            elif cur_sfc.startswith('yxz'):
                cur_ids = discrete_coords[:,[2,1,3]]
            elif cur_sfc.startswith('yzx'):
                cur_ids = discrete_coords[:,[2,3,1]]
            elif cur_sfc.startswith('zxy'):
                cur_ids = discrete_coords[:,[3,1,2]]
            elif cur_sfc.startswith('zyx'):
                cur_ids = discrete_coords[:,[3,2,1]]
                
            if cur_sfc.endswith('z-order'):  
                code = encode(cur_ids, batch=batch_ids, depth=16, order="z")
            elif cur_sfc.endswith('hilbert'):  
                code = encode(cur_ids, batch=batch_ids, depth=16, order="hilbert")
                
            codes.append(code)


        # Sort codes and create inverse.
        codes = torch.stack(codes)
        sorted_idxs = torch.argsort(codes)
        inv_sorted_idxs = torch.zeros_like(sorted_idxs).scatter_(
            dim=1,
            index=sorted_idxs,
            src=torch.arange(0, sorted_idxs.shape[1], device=sorted_idxs.device).repeat(
                sorted_idxs.shape[0], 1
            ),
        )


        return {
            'sparse_tensor': x,
            'cur_sfc': 0,
            'sfc_sorted_idxs': sorted_idxs,
            'sfc_inv_sorted_idxs': inv_sorted_idxs,
            'batch_offsets': batch_offsets,
            'max_counts': max_counts}


class SpaceFillingCurves2Sparse(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x['sparse_tensor']


class ResFormer(nn.Module):

    def __init__(self,
                 num_features,
                 num_heads,
                 window_size = 512,
                 path_drop_prob = 0.0,
                 att_drop_out = 0.0,
                 bias = True,
                 mlp_scale = 2,
                 norm_layer = None,
                 mlp_type = "standard"):
        
        super().__init__()

        self.window_size = window_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.att_drop_out = att_drop_out
        self.mlp_type = mlp_type
       
        # Att
        self.norm = norm_layer(num_features)
        self.linear_in = nn.Linear(num_features, num_features*3, bias = bias)
        self.linear_out = nn.Linear(num_features, num_features, bias = bias)
        self.skip_path_1 = SparseSkipConnection(path_drop_prob, num_features)

        # MLP
        if mlp_type == "standard":
            self.mlp = spconv.SparseSequential(*[
                norm_layer(num_features),
                nn.Linear(num_features, num_features*mlp_scale, bias = bias),
                nn.GELU(),
                nn.Linear(num_features*mlp_scale, num_features, bias = bias)])
        elif mlp_type == "glu":
            self.mlp_norm = norm_layer(num_features)
            self.mlp_lin_1 = nn.Linear(num_features, num_features*mlp_scale*2, bias = bias)
            self.mlp_lin_2 = nn.Linear(num_features*mlp_scale, num_features, bias = bias)
            self.mlp_act = nn.GELU()
        self.skip_path_2 = SparseSkipConnection(path_drop_prob, num_features)

    def forward(self, x):

        in_features = x['sparse_tensor']
        window = self.window_size
            
        ##### Multi-Head Attention
        
        # Space-filling curve idx selcetion.
        selected_sfc = x['cur_sfc']
        cur_idxs = x['sfc_sorted_idxs'][selected_sfc]
        cur_inv_idxs = x['sfc_inv_sorted_idxs'][selected_sfc]
        
        # Layer norm + projection qkv
        mid_features = self.norm(in_features)
        features = mid_features.features[cur_idxs]
        features = self.linear_in(features)
        features = features.reshape(
            (-1, 3, self.num_heads, self.num_features//self.num_heads))
        
        # Attention
        feat_dtype = features.dtype
        if feat_dtype != torch.float16:
            features = features.to(torch.float16)
        features = flash_attn_varlen_qkvpacked_func(
            features,
            x['batch_offsets'],
            x['max_counts'],
            dropout_p=self.att_drop_out,
            window_size=(window//2, window//2))\
                .reshape((-1, self.num_features))
        if feat_dtype != torch.float16:
            features = features.to(feat_dtype)
        
        # Projection out
        features = self.linear_out(features)

        # Sort back.
        features = features[cur_inv_idxs]
        mid_features = mid_features.replace_feature(features)

        # Skip path.
        mid_features = self.skip_path_1(mid_features, in_features)

        ##### F - Multi-Head Attention

        # ResNet MLP block.
        if self.mlp_type == "standard":
            end_features = self.mlp(mid_features)
        elif self.mlp_type == "glu":
            end_features = self.mlp_norm(mid_features)
            end_features_tensor_1, end_features_tensor_2 = self.mlp_lin_1(end_features.features).chunk(2, dim=-1)
            end_features_tensor = self.mlp_act(end_features_tensor_1) * end_features_tensor_2
            end_features_tensor = self.mlp_lin_2(end_features_tensor)
            end_features = end_features.replace_feature(end_features_tensor)
        end_features = self.skip_path_2(end_features, mid_features)
            
        # Compute sfc index for next layer.
        next_sfc = selected_sfc + 1
        if next_sfc >= len(x['sfc_sorted_idxs']):
            next_sfc = 0

        return {
            'sparse_tensor': end_features,
            'cur_sfc': next_sfc,
            'sfc_sorted_idxs' : x['sfc_sorted_idxs'],
            'sfc_inv_sorted_idxs': x['sfc_inv_sorted_idxs'],
            'batch_offsets' : x['batch_offsets'],
            'max_counts': x['max_counts']}
    