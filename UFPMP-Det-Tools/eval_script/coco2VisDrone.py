from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import json


def main():
    train_info = '/home/huangyecheng_2019/dataset/COCO/annotations/instances_UAVval_v1.json'
    # the coco result format json file 
    json_file = '/home/huangyecheng_2019/workspaces/mmdetection/UAV_bbox_result_tmp_v2.json'

    json_gt = json.load(open(train_info))

    json_result = json.load(open(json_file))

    img_id_2_img_name = {}
    img_id_2_img_results = {}
    for i in range(len(json_gt['images'])):
        name = json_gt['images'][i]['file_name'][:-4] + '.txt'
        id = json_gt['images'][i]['id']
        img_id_2_img_name[id] = name

    for item in json_result:
        bbox = item['bbox']
        score = item['score']
        image_id = item['image_id']
        category_id = item['category_id'] + 1
        if image_id not in img_id_2_img_results.keys():
            img_id_2_img_results[image_id] = []
        # if score < 0.1:
        #     continue
        img_id_2_img_results[image_id].append(
            [bbox[0], bbox[1], bbox[2], bbox[3], score, category_id, -1, -1]
        )

    root = '/home/huangyecheng_2019/workspaces/mmdetection/work_dirs/pred_txt/'

    for img_id in img_id_2_img_results.keys():
        name = img_id_2_img_name[img_id]
        fp = open(root+name,'w')
        scores = []
        for item in img_id_2_img_results[img_id]:
            # outline = ''
            scores.append(item[4])
        for idx in np.argsort(scores)[::-1]:
            item = img_id_2_img_results[img_id][idx]
            # print(item)
            outline = ''
            for num in item:
                outline += str(num) + ' '
            fp.write(outline + '\n')
        fp.close()
