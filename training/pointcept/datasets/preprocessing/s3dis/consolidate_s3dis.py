"""
Preprocessing Script for S3DIS
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import time
import torch
import subprocess
import shutil
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

def segmentator_parser(data_path, data_root, data_out):

    out_path = data_path.replace(data_root, data_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cur_pts = np.load(data_path+'/coord.npy')
    cur_normals = np.load(data_path+'/normal.npy')
    cur_rgb = np.load(data_path+'/color.npy')
    cur_labels = np.load(data_path+'/segment.npy')
    cur_obj_instance = np.load(data_path+'/instance.npy')
    
    # Save the segments.
    print(out_path+".pth")
    data_dict = {
        'coord': cur_pts.astype(np.float16),
        'color': cur_rgb.astype(np.uint8),
        'normal': cur_normals.astype(np.float16),
        'semantic_gt': cur_labels.astype(np.uint8), 
        'instance_gt': cur_obj_instance.astype(np.uint16)} 
    torch.save(data_dict, out_path+".pth")


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to processed S3DIS dataset"
    )
    parser.add_argument(
        "--dataset_out", required=True, help="Output path S3DIS dataset"
    )
    
    args = parser.parse_args()

    data_list = glob.glob(os.path.join(args.dataset_root, "Area*/*"))

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    _ = list(
        pool.map(
            segmentator_parser,
            data_list,
            repeat(args.dataset_root),
            repeat(args.dataset_out),
        )
    )


if __name__ == "__main__":
    main_process()
