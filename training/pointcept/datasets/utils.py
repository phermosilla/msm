"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list

    # 2D embeddings.
    if "emb_pt_idxs" in batch[0].keys():
        accum_num_pts = 0
        for i in range(len(batch)):
            batch[i]["emb_pt_idxs"] += accum_num_pts
            accum_num_pts += batch[i]["coord"].shape[0]

    # BB Idxs.
    if "bb_idxs" in batch[0].keys():
        accum_num_pts = 0
        for i in range(len(batch)):
            batch[i]["bb_idxs"] += accum_num_pts
            accum_num_pts += batch[i]["coord"].shape[0]

    # Default collate.
    batch = collate_fn(batch)

    # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
    if "offset" in batch.keys():
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0)

    # BB instance offset.
    #if 'bb_inst_offsets' in batch.keys():
    #    batch['bb_inst_offsets'] = torch.cumsum(batch['bb_inst_offsets'], dim=0)

    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
