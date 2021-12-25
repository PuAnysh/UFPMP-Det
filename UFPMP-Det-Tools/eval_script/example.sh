CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    ./configs/UFPMP-Det/mp_det_res50.py  \
    ./work_dirs/mp_det_res50/epoch_12.pth \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/UAVval
