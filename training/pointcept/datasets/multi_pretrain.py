"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class MultiPreTrainDataset(Dataset):

    def __init__(
        self,
        data_folders=[],
        pre_transform = None,
        clusters_file = None,
        loop = 1,
        transform=None
    ):
        super(MultiPreTrainDataset, self).__init__()
        self.transform = Compose(transform)
        self.pre_transform = Compose(pre_transform)
        self.loop = loop
        
        self.data_samples = []
        for cur_iter, cur_folder in enumerate(data_folders):
            self.data_samples = self.data_samples + self.get_data_list(cur_folder)

        if not clusters_file is None:
            self.cluster_dict = {}
            with open(clusters_file, 'r') as my_file:
                max_clusters = 0
                for cur_line in my_file:
                    split_line = cur_line.rstrip().split(',')
                    self.cluster_dict[split_line[0]] = int(split_line[1])
                    max_clusters = max(max_clusters, int(split_line[1]))

            self.cluster_list = [[] for _ in range(max_clusters+1)]
            for cur_iter, cur_file in enumerate(self.data_samples):
                key = os.path.basename(cur_file)[:-4]
                cluster_id = self.cluster_dict[key]
                self.cluster_list[cluster_id].append(cur_iter)

            new_cluster_list = []
            for cur_cluster_iter in range(len(self.cluster_list)):
                if len(self.cluster_list[cur_cluster_iter]) > 0:
                    new_cluster_list.append(self.cluster_list[cur_cluster_iter])
            self.cluster_list = new_cluster_list

            logger = get_root_logger()
            logger.info("Totally {} clusters.".format(len(self.cluster_list)))
            
        else:
            self.cluster_dict = None
            self.cluster_list = None

            logger = get_root_logger()
            logger.info("Totally {} samples.".format(len(self.data_samples)))


    def get_data_list(self, path):
        data_list = glob.glob(path + "*.pth")
        return [data_point for data_point in data_list]
    

    def get_data(self, idx):
        if self.cluster_list is None:
            data_path = self.data_samples[idx % len(self.data_samples)]
        else:
            file_idx = np.random.randint(0, len(self.cluster_list[idx  % len(self.cluster_list)]))
            data_path = self.data_samples[file_idx]
        data = torch.load(data_path)
        coord = data["coord"].astype(np.float32)
        color = data["color"].astype(np.float32)
        normal = data["normal"].astype(np.float32)
        normal = np.nan_to_num(normal)
        data_dict = dict(
            coord=coord,
            color=color,
            normal=normal,
            idx = idx % len(self.data_samples)
        )
        return data_dict


    def __getitem__(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.pre_transform(data_dict)

        data_dict_1 = dict()
        data_dict_2 = dict()
        for key in data_dict.keys():
            data_dict_1[key] = deepcopy(data_dict[key])
            data_dict_2[key] = deepcopy(data_dict[key])
        data_dict_1 = self.transform(data_dict_1)
        data_dict_2 = self.transform(data_dict_2)

        out_dict = {}
        for k, v in data_dict_1.items():
            out_dict['view_1_'+k] = v
        for k, v in data_dict_2.items():
            out_dict['view_2_'+k] = v
        out_dict['idx'] = data_dict['idx']
        return out_dict


    def __len__(self):
        if self.cluster_list is None:
            return len(self.data_samples) * self.loop
        else:
            return len(self.cluster_list) * self.loop
