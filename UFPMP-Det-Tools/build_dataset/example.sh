CUDA_VISIBLE_DEVICES=7 python UFPMP-Det-Tools/build_dataset/UFP_VisDrone2COCO.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    /home/huangyecheng_2019/dataset/COCO/images/UAVtrain \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVtrain_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/instance_UFP_UAVtrain_test/ \
    /home/huangyecheng_2019/dataset/COCO/annotations/instance_UFP_UAVtrain_test.json 


CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/build_dataset/UFP_VisDrone2COCO.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    /home/huangyecheng_2019/dataset/COCO/images/UAVval \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/instance_UFP_UAVval/ \
    /home/huangyecheng_2019/dataset/COCO/annotations/instance_UFP_UAVval.json


CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/build_dataset/UFP_VisDrone2COCO.py \
    ./configs/UFPMP-Det/UAVDT_coarse_det.py \
    ./work_dirs/UAVDT_coarse_det/epoch_11.pth \
    /home/huangyecheng_2019/dataset/COCO/images/UAVval \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/instance_UFP_UAVval/ \
    /home/huangyecheng_2019/dataset/COCO/annotations/instance_UFP_UAVval.json