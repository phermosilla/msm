import copy
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv
#from torch_geometric.utils import scatter
from ..builder import MODELS, build_model
from pointcept.models.utils import offset2batch, batch2offset
from .cross_modal_module import MatchModule

@MODELS.register_module()
class ObjectGrounding(nn.Module):
    def __init__(self, 
                 backbone = None,
                 out_bb_dim = 64,
                 in_feat_dim = 128,
                 max_bb = 110):
        super().__init__()

        self.backbone = build_model(backbone)
        self.clip, _ = clip.load("ViT-B/32")
        for param in self.clip.parameters():
            param.requires_grad = False

        self.max_bb = max_bb

        self.word_transform = nn.Sequential(
            nn.Linear(self.clip.visual.output_dim, in_feat_dim),
            nn.LayerNorm(in_feat_dim, eps=1e-4),
            nn.GELU(),
            nn.Linear(in_feat_dim, in_feat_dim))
        
        self.bb_transform = nn.Sequential(
            nn.Linear(out_bb_dim, in_feat_dim),
            nn.LayerNorm(in_feat_dim, eps=1e-4),
            nn.GELU(),
            nn.Linear(in_feat_dim, in_feat_dim))
        
        self.match_module = MatchModule(in_feat_dim, 4, 2)

        self.loss = nn.CrossEntropyLoss()
        

    def get_clip_word_features(self, text_emb):
        # [batch_size, n_ctx, d_model]
        x = self.clip.token_embedding(text_emb).type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return self.clip.ln_final(x).type(self.clip.dtype)
    

    def forward(self, input_dict):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                # Get text features.
                text_feats = self.get_clip_word_features(input_dict['text_emb'])
                text_mask = input_dict['text_emb'] > 0

            # Get point features.
            pt_features = self.backbone(input_dict)

            # Mean feature bb.
            batch_instances = offset2batch(input_dict['bb_inst_offsets'])
            sel_feats = pt_features[input_dict['bb_idxs'],:]

            indices = batch_instances.unsqueeze(1).repeat((1, sel_feats.shape[1]))
            bb_feats = torch.zeros(
                (input_dict['bb_inst_offsets'].shape[0], sel_feats.shape[1]), 
                dtype=sel_feats.dtype, device=sel_feats.device)
            bb_feats.scatter_add_(0, indices, sel_feats)
            with torch.no_grad():
                bb_counts = torch.zeros_like(bb_feats)
                bb_counts_src = torch.ones_like(sel_feats)
                bb_counts.scatter_add_(0, indices, bb_counts_src)
                bb_counts = torch.clamp(bb_counts, min=1)
            bb_feats = bb_feats/bb_counts
            
            #bb_feats = scatter(
            #    sel_feats,
            #    batch_instances.to(torch.int64),
            #    dim = 0, 
            #    dim_size = input_dict['bb_inst_offsets'].shape[0], 
            #    reduce = 'mean')

            bb_feats = bb_feats.reshape((-1, self.max_bb, sel_feats.shape[-1]))
            bb_centers = input_dict['bb_centers'].reshape((-1, self.max_bb, 3))
            bb_mask = input_dict['bb_mask'].reshape((-1, self.max_bb))
            
            # Process text.
            text_feats = self.word_transform(text_feats)

            # Process bb features.
            bb_feats = self.bb_transform(bb_feats)

            # Invert masks.
            bb_mask = torch.logical_not(bb_mask)
            text_mask = torch.logical_not(text_mask)

            # Match module.
            scores = self.match_module(
                text_feats=text_feats, 
                text_mask=text_mask, 
                bb_feats=bb_feats, 
                bb_mask=bb_mask, 
                bb_centers=bb_centers)
            scores = scores.masked_fill(bb_mask, -torch.inf)
            
            # Loss.
            loss = self.loss(scores, input_dict['sel_obj'])

        # Accuracy.
        acc = torch.mean((scores.argmax(1) == input_dict['sel_obj']).to(torch.float32))

        ######## Train
        if self.training:
            out_dict = dict(
                loss=loss,
                acc=acc*100.
            )
        else:
            out_dict = dict(
                scores=scores,
                loss=loss,
                acc=acc*100.
            )

        # Return.
        return out_dict
