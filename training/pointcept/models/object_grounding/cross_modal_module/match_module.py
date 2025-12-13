import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class MatchModule(nn.Module):
    def __init__(self, feat_channel, head, depth):
        super().__init__()
        self.depth = depth
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=feat_channel,
                h=head,
                d_k=feat_channel // head,
                d_v=feat_channel // head,
                dropout=0.1
            ) for _ in range(depth)
        )
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=feat_channel,
                h=head,
                d_k=feat_channel // head,
                d_v=feat_channel // head,
                dropout=0.1
            ) for _ in range(depth)
        )
        self.match = nn.Sequential(
            nn.Linear(feat_channel, feat_channel),
            nn.LayerNorm(feat_channel, eps=1e-4),
            nn.GELU(),
            nn.Linear(feat_channel, feat_channel),
            nn.LayerNorm(feat_channel, eps=1e-4),
            nn.GELU(),
            nn.Linear(feat_channel, 1)
        )

    def forward(self, text_feats, text_mask, bb_feats, bb_mask, bb_centers):
        
        # Get dimensions.
        batch_size = text_feats.shape[0]
        max_words = text_feats.shape[1]
        max_bb = bb_feats.shape[1]

        # Attention weight.
        attention_weights = self._calculate_spatial_weight(bb_mask, bb_centers).to(bb_feats.dtype)

        # Mask text.
        tmp_unsqueezed = text_mask.unsqueeze(-1).to(torch.float32)
        tmp_unsqueezed_bb = bb_mask.unsqueeze(-1).to(torch.float32)
        text_mask = (tmp_unsqueezed.transpose(1, 2) * tmp_unsqueezed_bb).unsqueeze(dim=1)
        text_mask = text_mask.to(bool)
        
        # Mask bounding box.
        bb_mask = (tmp_unsqueezed_bb.transpose(1, 2) * tmp_unsqueezed_bb).unsqueeze(dim=1)
        bb_mask = bb_mask.to(bool)
        
        # Compute cross attention.
        for i in range(self.depth):
            bb_feats = self.self_attn[i](
                bb_feats, 
                bb_feats, 
                bb_feats,
                attention_mask=bb_mask, 
                attention_weights=attention_weights, 
                way="add")
            bb_feats = self.cross_attn[i](
                bb_feats, 
                text_feats, 
                text_feats,
                attention_mask=text_mask)
        # match
        return self.match(bb_feats).reshape(batch_size, max_bb)
    

    def _calculate_spatial_weight(self, bb_mask, bb_centers):
        """
        Reference: https://github.com/zlccccc/3DVG-Transformer
        """
        # Compute pair distances.
        dist = (bb_centers.unsqueeze(dim=1) - bb_centers.unsqueeze(dim=2)).pow(2)
        dist = torch.sqrt(dist.sum(dim=-1)).unsqueeze(dim=1)
        dist_weights = 1 / (dist + 1e-2) # N x 1 x BB x BB

        # Apply mask.
        tmp_unsqueezed = bb_mask.unsqueeze(-1).to(torch.float32)
        dist_weights *= (tmp_unsqueezed.transpose(1, 2) * tmp_unsqueezed).unsqueeze(dim=1)

        # Normalize.
        dist_weights += torch.finfo(torch.float32).eps  # prevent zeros
        norm = dist_weights.sum(dim=2, keepdim=True)
        dist_weights = dist_weights / norm
        zeros = torch.zeros_like(dist_weights)
        dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
        return dist_weights
