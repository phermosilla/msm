# Masked Scene Modeling

[Pedro Hermosilla](https://phermosilla.github.io) and [Christian Stippel](https://scholar.google.at/citations?user=Vf9eONQAAAAJ&hl=en) and [Leon Sick](https://leonsick.github.io/)

This is the official repository of the **CVPR 2025** paper **Masked Scene Modeling: Narrowing the Gap Between Supervised and Self-Supervised Learning in 3D Scene Understanding**, and it will be mainly used to share the inference and pre-training code, as well as the pre-trained weights for the models described in our paper.

[[Arxiv](https://arxiv.org/abs/2504.06719)] - [[Project](https://phermosilla.github.io/msm)]

**Self-Supervised Feature Visualization using PCA**
![Teaser](/imgs/teaser.png)

### Abstract
*Self-supervised learning has transformed 2D computer vision by enabling models trained on large, unannotated datasets to provide versatile off-the-shelf features that perform similarly to models trained with labels. However, in 3D scene understanding, self-supervised methods are typically only used as a weight initialization step for task-specific fine-tuning, limiting their utility for general-purpose feature extraction.  This paper addresses this shortcoming by proposing a robust evaluation protocol specifically designed to assess the quality of self-supervised features for 3D scene understanding. Our protocol uses multi-resolution feature sampling of hierarchical models to create rich point-level representations that capture the semantic capabilities of the model and, hence, are suitable for evaluation with linear probing and nearest-neighbor methods. Furthermore, we introduce the first self-supervised model that performs similarly to supervised models when only off-the-shelf features are used in a linear probing setup.In particular, our model is trained natively in 3D with a novel self-supervised approach based on a Masked Scene Modeling objective, which reconstructs deep features of masked patches in a bottom-up manner and is specifically tailored to hierarchical 3D models. Our experiments not only demonstrate that our method achieves competitive performance to supervised models, but also surpasses existing self-supervised approaches by a large margin.*

## Project structure
We provide code and instructions for the inference and visualization of our model:

```
msm
├── inference/
│   ├── model/
│   ├── transform/
│   ├── utils/
│   ├── README.md
│   ├── dependencies.sh
│   ├── inference.py
│   ├── model_cfg.py
│   ├── preprocessor.py
│   └── requirements.txt
├── visualization/
│   ├── README.md
│   ├── blender_load_colored_pc.py
│   ├── blender_template.blend
│   ├── create_pca.py
│   └── LICENSE
└── README.md
```


### Inference
Our model can generate **semantic off-the-shelf feature embeddings for any 3D scene**.
For this, we provide an installation manual and an inference script that works out of the box.
To generate embeddings for your scene using our pre-trained model, please refer to the [inference folder](/inference/).

### Embedding visualization
![](/imgs/visualization.png)
We also provide the code used to generate the scene visualizations in our paper. 
If you want to visualize your point embeddings using PCA, follow the instructions in the [visualization folder](/visualization/).


### Schedule
Future Releases:

- [x] ~~Inference model code~~
- [x] ~~Pre-trained weights~~
- [ ] Pre-training code
- [ ] Pre-training config files
- [ ] Config files ablation studies
- [x] ~~PCA feature visualization code~~

## Citation

If you find our work useful to your research, please cite our work as an acknowledgment.
```bib
@inproceedings{hermosilla2025msm,
    title={Masked Scene Modeling: Narrowing the Gap Between Supervised and Self-Supervised Learning in 3D Scene Understanding}, 
    author={Hermosilla, Pedro and Stippel, Christian and Sick, Leon},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}