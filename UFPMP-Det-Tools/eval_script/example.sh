CUDA_VISIBLE_DEVICES=4 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    ./configs/UFPMP-Det/mp_det_res50.py  \
    ./work_dirs/mp_det_res50/epoch_12.pth \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/UAVval

CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/UAVDT_coarse_det.py \
    ./work_dirs/UAVDT_coarse_det/epoch_11.pth \
    ./configs/UFPMP-Det/uavdt_mp_det_res50.py  \
    ./work_dirs/uavdt_mp_det_res50/epoch_8.pth \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVDTval.json \
    /home/huangyecheng_2019/dataset/COCO/images/UAV-benchmark-M

CUDA_VISIBLE_DEVICES=1 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    /home/huangyecheng_2019/workspaces/mmdetection/work_dirs/fsaf_uav_dt_los_res/fsaf_uav_dt_los_res.py \
    /home/huangyecheng_2019/workspaces/mmdetection/work_dirs/fsaf_uav_dt_los_res/epoch_3.pth \
    ./configs/UFPMP-Det/uavdt_mp_det_res50.py  \
    ./work_dirs/uavdt_mp_det_res50/epoch_8.pth \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVDTval.json \
    /home/huangyecheng_2019/dataset/COCO//images/UAV_with_mask/


CUDA_VISIBLE_DEVICES=4 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    ./configs/UFPMP-Det/mp_det_res101.py  \
    ./work_dirs/mp_det_res101/epoch_12.pth \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/UAVval