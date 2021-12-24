CUDA_VISIBLE_DEVICES=2 python UFPMP-Det-Tools/eval_script/ufpmp_det_eval.py \
    ./configs/UFPMP-Det/coarse_det.py \
    ./work_dirs/coarse_det/epoch_12.pth \
    /home/huangyecheng_2019/workspaces/mmdetection/configs/UAV/glf_baseline_res50.py \
    /home/huangyecheng_2019/workspaces/mmdetection/work_dirs/glf_baseline_res50/epoch_22.pth \
    /home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json \
    /home/huangyecheng_2019/dataset/COCO/images/UAVval
