# UFPMP-Det: Toward Accurate and Efficient Object Detection on Drone Imagery

The repo is the official implement of  UFPMP-Det.

The code of **UFP** **module** is at [mmdet/core/ufp](mmdet/core/ufp)

The code of **MP-Det is** at [mmdet/models/dense_heads/mp_head.py](mmdet/models/dense_heads/mp_head.py)

The **config** of our project is at [configs/UFPMP-Det](configs/UFPMP-Det)

# Install

1. This repo is implemented based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please install it according to [get_start.md](docs/en/get_started.md).
2. ```shell
   pip install nltk
   pip install albumentations
   ```
## Quick start
We provide the Dataset(COCO Format) as follow:
- VisDrone:链接：https://pan.baidu.com/s/1FfAsAApHZruucO5A2QgQAg 提取码：qrvs
- UAVDT:链接：链接：https://pan.baidu.com/s/1KLmU5BBWwgtFbuZa7QWavw 提取码：z08x

We provide the checkpoint as follow:
coming soon

# Training

This repo only suppose singal GPU.

## Prepare

Build by yourself: We provide two data set conversion tools.

```shell
# conver VisDrone to COCO
python UFPMP-Det-Tools/build_dataset/VisDrone2COCO.py
# conver UAVDT to COCO
python UFPMP-Det-Tools/build_dataset/UAVDT2COCO.py
# build UFP dataset(VisDrone)
CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/build_dataset/UFP_VisDrone2COCO.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    xxxxxx/dataset/COCO/images/UAVtrain \
    xxxxxx/dataset/COCO/annotations/instances_UAVtrain_v1.json \
    xxxxxx/dataset/COCO/images/instance_UFP_UAVtrain/ \
    xxxxxx/dataset/COCO/annotations/instance_UFP_UAVtrain.json \
    --txt_path path_to_VisDrone_annotation_dir
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
CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    ./configs/UFPMP-Det/mp_det_res50.py  \
    ./work_dirs/mp_det_res50/epoch_12.pth \
    XXXXX/dataset/COCO/annotations/instances_UAVval_v1.json \
    XXXXX/dataset/COCO/images/UAVval

```
