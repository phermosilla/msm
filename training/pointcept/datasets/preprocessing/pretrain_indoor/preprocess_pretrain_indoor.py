import numpy as np
import torch
import glob
import os

from pointcept.datasets.transform import GridSample

def voxelize_parser(in_data, voxel_size):
    data = GridSample(
        grid_size=voxel_size, 
        hash_type="fnv", 
        mode="train", 
        keys=in_data.keys())(in_data)
    return data


out_path = "./data/pretrain_indoor_1cm/"
data_folders=[
    "./data/scannet_data/train/",
    "./data/scannet_data/test/",
    "./data/scannet_data/val/"]
prefixes = [
    "scannet",
    "scannet",
    "scannet"]
voxel_size = 0.01
voxelize = [
    False, 
    False, 
    False]
dict_format = [
    True, 
    True, 
    True]

file_list = []
file_list_prefixes = []
file_list_voxelize = []
file_list_dict_format = []
for cur_folder, cur_prefix, cur_voxelize, cur_dict_format in zip(data_folders, prefixes, voxelize, dict_format):
    data_list = glob.glob(os.path.join(cur_folder, "*.pth"))
    file_list = file_list + data_list
    for cur_file in data_list:
        cur_file_prefix = cur_prefix + "_" + os.path.basename(os.path.dirname(cur_file))
        file_list_prefixes.append(cur_file_prefix)   
        file_list_voxelize.append(cur_voxelize) 
        file_list_dict_format.append(cur_dict_format)

cur_iter = 0
prev_file_sizes = []
new_file_sizes = []
for cur_file, cur_prefix, cur_voxelize, cur_dict_format in \
    zip(file_list, file_list_prefixes, file_list_voxelize, file_list_dict_format):
    file_base_name = cur_prefix+"_"+os.path.basename(cur_file)
    cur_out_path = os.path.join(out_path, file_base_name)
    
    if cur_iter % 500 == 0: 
        print("{:5d} / {:5d}: {:.2f} {:.2f} {:s} {:s}".format(
            cur_iter, len(file_list), np.mean(prev_file_sizes), 
            np.mean(new_file_sizes), cur_prefix, cur_file))
        
    if not os.path.isfile(cur_out_path):
        cur_data = torch.load(cur_file)
        if cur_voxelize:
            cur_data = {
                "coord" : cur_data["coord"],
                "color" : cur_data["color"],
                "normal" : cur_data["normal"]}
            cur_data = voxelize_parser(cur_data, voxel_size)
        if cur_dict_format:
            new_dict = {
                "coord" : cur_data["coord"].astype(np.float16),
                "color" : cur_data["color"].astype(np.uint8),
                "normal" : cur_data["normal"].astype(np.float16)}
        else:
            cur_pts = cur_data[0]
            cur_normals = cur_data[1]
            cur_rgb = ((cur_data[2] + 1.0)*127.5).astype(np.uint8)
            new_dict = {
                "coord" : cur_pts.astype(np.float16),
                "color" : cur_rgb.astype(np.uint8),
                "normal" : cur_normals.astype(np.float16)}
        torch.save(new_dict, cur_out_path)

    prev_file_sizes.append(os.path.getsize(cur_file)/1048576)
    new_file_sizes.append(os.path.getsize(cur_out_path)/1048576)

    cur_iter += 1

print(len(file_list))