# Training

## Setup environment

For simplicity, we use docker to train and evaluate our models. First, we run the docker image:

```
docker run --rm --shm-size=32gb --gpus all --privileged -it -v ${PWD}:/working_dir -w /working_dir pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```

Once the docker image is running, we install dependencies:

```
sh dependencies.sh
export PYTHONPATH=./
```

Lastly, we configure wandb with our login key
```
wandb login [KEY]
```
or if we do not want to use wandb, we can set up the offline mode instead.
```
wandb offline
```

## Pre-process data

### Scannet dataset

Download the scannet dataset from the official repository and create the output directory at <code>data/scannet_data/</code>. If you want to change the destination folder make sure to modify the path in the config files for the different experiments. Then, run the following command:

```
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root [SOURCE_SCANNET_PATH] --output_root data/scannet_data/
```

### S3DIS dataset

First, create the empty dataset folder at <code>data/s3dis/</code>. Then, download the oversegmentation ids from the following [link](https://drive.google.com/file/d/1M4kMVLvLDe0D123lcmPtUxbWrLlVH4B5/view?usp=sharing) and copy the <code>Area_*</code> folders into <code>data/s3dis/</code>. Then, download the pre-processed data from Pointcept from the following [link](https://huggingface.co/datasets/Pointcept/s3dis-compressed) at any folder. Lastly, convert the Poincept dataset format into pth format using the following command:

```
python pointcept/datasets/preprocessing/s3dis/consolidate_s3dis.py --dataset_root [POINTCEPT_S3DIS_PATH] --dataset_out data/s3dis
```

### Scanrefer dataset

First, download the scanrefer dataset from the following [link](https://github.com/daveredrum/ScanRefer). Then, process the dataset with the following command:

```
python pointcept/datasets/preprocessing/scanrefer/process_scanrefer.py --dataset_root [SCANREFER_PATH] --output_root data/scanrefer
```

### Self-supervised dataset

First, create the output folder at <code>"data/pretrain_indoor_1cm"</code>. Then, execute the following command to transform the scannet dataset into the format for self-supervised training:

```
python pointcept/datasets/preprocessing/pretrain_indoor/preprocess_pretrain_indoor.py
```

If you want to add additional datasets for self-supervised pre-training, add then into the list within the <code>preprocess_pretrain_indoor.py</code> file.

## Training self-supervised model

To train the self-supervised model, we only need to execute the following command (adapt the number of gpus of your system, but the model should be able to train on 2 x A6000 GPUs with 40 Gb each):

```
python tools/train.py --config-file configs/msm/msm_scannet.py --num-gpus 2 --options save_path=logs/msm_scannet
```


## Evaluation

Once the model is trained, we can evaluate the performance of the model on different tasks. You would need to modify the path to the model weights on the file <code>configs/msm/msm_model.py</code>.

### Semantic segmentation

We can evaluate our pre-trained model on the task of semantic segmentation using nearest-neighbors protocol with the following commands:

```
python tools/test.py --config-file configs/msm/eval/nn-scannet.py --options save_path=log_nn_scannet weight=None
python tools/test.py --config-file configs/msm/eval/nn-scannet200.py --options save_path=log_nn_scannet200 weight=None
python tools/test.py --config-file configs/msm/eval/nn-s3dis.py --options save_path=log_nn_s3dis weight=None
```

We can evaluate our pre-trained model on the task of semantic segmentation using linear probing protocol with the following command:

```
python tools/train.py --config-file configs/msm/eval/linear-scannet.py --num-gpus 2 --options save_path=log_lin_scannet
python tools/train.py --config-file configs/msm/eval/linear-scannet200.py --num-gpus 2 --options save_path=log_lin_scannet200
python tools/train.py --config-file configs/msm/eval/linear-s3dis.py --num-gpus 2 --options save_path=log_lin_s3dis
```

### Semantic segmentation (Efficiency benchmark)

On the config files <code>configs/msm/eval/nn-scannet.py</code> and <code>configs/msm/eval/linear-scannet.py</code>, substitute the <code>lr_file</code> and <code>la_file</code> variables to activate the different efficiency benchmarks. The evaluation is executed with the same commands as in Semantic Segmentation experiments.

### Instance segmentation

We can evaluate our pre-trained model on the task of instance segmentation using following commands:

```
python tools/train.py --config-file configs/msm/eval/linear-scannet-inst.py --num-gpus 2 --options save_path=log_inst_scannet
python tools/train.py --config-file configs/msm/eval/linear-scannet200-inst.py --num-gpus 2 --options save_path=log_inst_scannet200
python tools/train.py --config-file configs/msm/eval/linear-s3dis-inst.py --num-gpus 2 --options save_path=log_inst_s3dis
```

### 3D Visual Grounding

We can evaluate our pre-trained model on the task of 3d visual grounding using following commands:

```
python tools/train.py --config-file configs/msm/eval/og-scannet.py --num-gpus 2 --options save_path=log_og_scanrefer
```

## Acknowledgement

This project is based on the code-base [Pointcept](https://github.com/Pointcept/Pointcept). 