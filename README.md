# UFPMP-Det: Toward Accurate and Efficient Object Detection on Drone Imagery

The repo is the official implement of  UFPMP-Det.

The code of **UFP** **module** is at [mmdet/core/ufp](mmdet/core/ufp)

The code of MP-Det is at [mmdet/models/dense_heads/mp_head.py](mmdet/models/dense_heads/mp_head.py)

The config of our project is at [configs/UFPMP-Det](configs/UFPMP-Det)

# Install

1. This repo is implemented based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please install it according to [get_start.md](docs/en/get_started.md).
2. ```shell
   pip install nltk
   pip install albumentations
   ```
## Quick start
We provide the checkport as follow:

# Training

This repo only suppose singal GPU.

## Prepare

Build by yourself: We provide two data set conversion tools.

```shell
# conver VisDrone to COCO
python UFPMP-Det-Tools/build_dataset/VisDrone2COCO.py
# conver UAVDT to COCO
python UFPMP-Det-Tools/build_dataset/UAVDT2COCO.py
```

Download:

coming soon

## Train Coarse Detector

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/UFPMP-Det/coarse_det.py
```

## Train MP-Det

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ./config/UFPMP-Det/mp_det_res50.py
```

# Test

```shell
CUDA_VISIBLE_DEVICES=0 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py

```
