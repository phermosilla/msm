"""
This help file adds evaluation labels (unique / multiple) to original ScanRefer data, it follows the
official code logic. Please refer to
https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py
"""

from tqdm.contrib.concurrent import process_map
from functools import partial
import json
import csv
import argparse
import shutil

def get_semantic_mapping_file(file_path, mapping_name):
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[mapping_col_idx[mapping_name]])
    return label_mapping


def add_unique_multiple_labels_to_json(file_path, output_path, label_mapping, valid_semantic_mapping):
    with open(file_path, "r") as f:
        scanrefer_json_data = json.load(f)
    obj_cache = {}
    sem_cache = {}
    for item in scanrefer_json_data:
        if (item["scene_id"], item["object_id"]) in obj_cache:
            continue
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        if (item['scene_id'], sem_label) not in sem_cache:
            sem_cache[(item['scene_id'], sem_label)] = 0
        sem_cache[(item['scene_id'], sem_label)] += 1
        obj_cache[(item["scene_id"], item["object_id"])] = True

    for item in scanrefer_json_data:
        scene_id = item['scene_id']
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        assert sem_cache[(scene_id, sem_label)] >= 1
        item["eval_type"] = "unique" if sem_cache[(scene_id, sem_label)] == 1 else "multiple"
    
    # save the new json
    with open(output_path, "w") as f:
        json.dump(scanrefer_json_data, f, indent=2)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Scanrefer dataset",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path",
    )
    config = parser.parse_args()

    print("\nDefault: using all CPU cores.")
    label_mapping = get_semantic_mapping_file("pointcept/datasets/preprocessing/scanrefer/meta_data/scannetv2-labels.combined.tsv", "nyu40")
    for split in ("train", "val"):
        add_unique_multiple_labels_to_json(
            config.dataset_root+f"/ScanRefer_filtered_{split}.json",
            config.output_root+f"/ScanRefer_filtered_{split}.json",
            label_mapping=label_mapping,
            valid_semantic_mapping=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        
    shutil.copyfile("pointcept/datasets/preprocessing/scanrefer/meta_data/ScanRefer_filtered_train.txt", config.output_root+"/ScanRefer_filtered_train.txt")
    shutil.copyfile("pointcept/datasets/preprocessing/scanrefer/meta_data/ScanRefer_filtered_val.txt", config.output_root+"/ScanRefer_filtered_val.txt")
    print(f"==> Complete.")