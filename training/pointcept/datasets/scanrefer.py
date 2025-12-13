"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import json
import clip
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanReferDataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        split="train",
        data_root="data/scanrefer",
        data_root_scannet="data/scannet",
        max_bb = 110,
        filter = None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super(ScanReferDataset, self).__init__()
        self.data_root = data_root
        self.data_root_scannet = data_root_scannet
        self.split = split
        self.max_bb = max_bb
        self.filter = filter
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()

        if not self.filter is None:
            new_data_list = []
            for cur_elem in self.data_list:
                if cur_elem['eval_type'] == self.filter:
                    new_data_list.append(cur_elem)
            self.data_list = new_data_list

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )


    def get_data_list(self):
        file_path = os.path.join(self.data_root, "ScanRefer_filtered_"+self.split+".json")
        with open(file_path, "r") as f:
            scanrefer_json_data = json.load(f)
        return scanrefer_json_data
    

    def compute_bb(self, coords, instances):
        unique_idxs = torch.unique(instances, sorted=True)
        unique_idxs = unique_idxs[unique_idxs != -1]
        
        bb_idxs = []
        bb_centers = []
        bb_inst_offset = []
        bb_mask = []
        for cur_idxs in unique_idxs:
            mask = instances == cur_idxs
            bb_min = torch.amin(coords[mask], 0)
            bb_max = torch.amax(coords[mask], 0)
            bb_centers.append((bb_max + bb_min)*0.5)
            mask_max = (coords[:,0] < bb_max[0]) & (coords[:,1] < bb_max[1]) & (coords[:,2] < bb_max[2])
            mask_min = (coords[:,0] > bb_min[0]) & (coords[:,1] > bb_min[1]) & (coords[:,2] > bb_min[2])
            bb_idxs.append((mask_max & mask_min).nonzero(as_tuple=True)[0])
            bb_inst_offset.append(bb_idxs[-1].shape[0])
            bb_mask.append(True)
        
        for i in range(unique_idxs.shape[0], self.max_bb):
            bb_centers.append(torch.from_numpy(np.array([.0, .0, .0])))
            bb_inst_offset.append(0)
            bb_mask.append(False)

        return unique_idxs, torch.stack(bb_centers), torch.cat(bb_idxs), \
            torch.from_numpy(np.array(bb_inst_offset)), \
            torch.from_numpy(np.array(bb_mask))


    def get_data(self, idx):
        json_data = self.data_list[idx % len(self.data_list)]
        scene_path = os.path.join(self.data_root_scannet, self.split, json_data['scene_id']+".pth")

        # Get the scannet data.
        scene_data = torch.load(scene_path)        
        coord = scene_data["coord"]
        color = scene_data["color"]
        normal = np.nan_to_num(scene_data["normal"])
        instance = scene_data["instance_gt"].reshape([-1])

        # Return processed data.
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            instance=instance)
        return data_dict


    def get_data_text(self, idx, data_dict):
        json_data = self.data_list[idx % len(self.data_list)]

        # Get original data.
        coord = data_dict['coord']
        instance = data_dict['instance']
        grid_coord = data_dict['grid_coord']
        feat = data_dict['feat']
        offset = data_dict['offset']
        displacement = data_dict['displacement']

        # Get scanrefer data.
        scene_id = json_data['scene_id']
        sel_obj = int(json_data['object_id'])
        with torch.no_grad():
            text_emb = clip.tokenize(json_data["description"].strip(), truncate=True)
            
        # Compute bb idxs.
        _, bb_centers, bb_idxs, bb_inst_offsets, bb_mask = self.compute_bb(coord, instance)
        
        # Return processed data.
        data_dict = dict(
            coord=coord,
            grid_coord=grid_coord,
            feat=feat,
            offset=offset,
            displacement=displacement,
            name=scene_id,
            sel_obj=sel_obj,
            text_emb=text_emb, 
            bb_centers=bb_centers, 
            bb_idxs=bb_idxs, 
            bb_inst_offsets=bb_inst_offsets,
            bb_mask=bb_mask)
        return data_dict
    

    def get_data_name(self, idx):
        json_data = self.data_list[idx % len(self.data_list)]
        return json_data['scene_id']
    

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        data_dict = self.get_data_text(idx, data_dict)
        return data_dict
    

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            input_dict_list += [data_part_list]

        for i in range(len(input_dict_list)):
            cur_dict = self.post_transform(input_dict_list[i])
            input_dict_list[i] = self.get_data_text(idx, cur_dict)
        
        data_dict = dict(fragment_list=input_dict_list, name=self.get_data_name(idx))
        return data_dict
    

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)
        

    def __len__(self):
        return len(self.data_list) * self.loop

#transform=[
#    dict(type="CenterShift", apply_z=True),
#    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
#    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
#    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
#    dict(type="RandomScale", scale=[0.9, 1.1]),
#    dict(type="RandomFlip", p=0.5),
#    dict(type="RandomJitter", sigma=0.005, clip=0.02),
#    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
#    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
#    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
#    dict(type="ChromaticJitter", p=0.95, std=0.05),
#    dict(
#        type="GridSample",
#        grid_size=0.02,
#        hash_type="fnv",
#        mode="train",
#        keys=("coord", "color", "normal", "instance"),
#        return_grid_coord=True,
#        return_min_coord=True,
#        return_displacement=True,
#        project_displacement=False,
#    ),
#    dict(type="NormalizeColor"),
#    dict(type="ToTensor"),
#    dict(
#        type="Collect",
#        keys=("coord", "grid_coord", "displacement", "instance"),
#        feat_keys=("color", "normal"),
#    )]
#dataset = ScanReferDataset(
#        split="train",
#        data_root="data/scanrefer",
#        data_root_scannet="data/scannet_data",
#        transform=transform,
#        test_mode=False,
#        test_cfg=None,
#        cache=False,
#        loop=1)
#cur_item = dataset.__getitem__(3000)
#coords = cur_item['coord'].numpy()
#color = np.zeros_like(coords)
#bb_idxs = cur_item['bb_idxs'].numpy()
#bb_offsets = cur_item['bb_inst_offsets'].numpy()
#bb_offsets = np.cumsum(bb_offsets)
#bb_centers = cur_item['bb_centers'].numpy()
#sel_obj = cur_item['sel_obj']-1
#print()
#print(cur_item['name'])
#print(cur_item['text_emb'])
#print(sel_obj)
#print()
#color[bb_idxs[bb_offsets[sel_obj]:bb_offsets[sel_obj+1]], :] = np.array([1.0, 0.0, 0.0])
#np.savetxt("example_scene.txt", np.concatenate((coords, color), -1))