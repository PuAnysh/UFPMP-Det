from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import warnings
import cv2
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.ops import RoIAlign, RoIPool
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
from analyse.merge_bbox import merge_bbox
from mmdet.core import UnifiedForegroundPacking
import math
import copy


def compute_iof(pos1, pos2):
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    # 计算中间重叠区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    else:
        inter = (right - left) * (bottom - top)
        return inter / min(area1, area2)


# modify test
class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results, bbox=None):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        if bbox:
            x1, x2, y1, y2, _ = bbox
            img = img[x1:x2, y1:y2, :]
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def my_inference_detector(model, data):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def merge_result(final_result, second_result, offset):
    offset_x, offset_y = offset
    for idx, result in enumerate(second_result):
        result[:, 0] += offset_x
        result[:, 1] += offset_y
        result[:, 2] += offset_x
        result[:, 3] += offset_y
        final_result[idx] = np.vstack((final_result[idx], result))
    return final_result


def display_result(results, img, img_name):
    img_data = cv2.imread(img)
    for result in results:
        x1, y1, w, h, n_x, n_y = result
        cv2.rectangle(img_data, (x1, y1), (x1 + w, y1 + h), (0, 0, 255))
    # cv2.imwrite('/home/huangyecheng/UAV/mmdetection/work_dirs/img_results/' + img_name, img_data)


def display_result_gt(results, img, img_name):
    img_data = cv2.imread(img)
    for result in results:
        x1, y1, w, h = result
        cv2.rectangle(img_data, (x1, y1), (x1 + w, y1 + h), (0, 0, 255))
    # cv2.imwrite('/home/huangyecheng/UAV/mmdetection/work_dirs/img_results/' + img_name, img_data)


def display_merge_result(results, img, img_name, w, h):
    w = math.ceil(w)
    h = math.ceil(h)
    img_data = cv2.imread(img)
    anno_path = '/home/huangyecheng/dataset/COCO/annotations/UAVtrain/' + img_name[:-4] + '.txt'
    anno = open(anno_path)
    for rec in anno.readlines():
        # print(rec,img_name)
        rec = rec.strip()
        x, y, _w, _h, s, _cls = [int(_) for _ in rec.split(',')[:6]]
        if _cls == 0:
            img_data[y:y + _h, x:x + _w, :] = 0
    anno.close()
    new_img = np.zeros((h, w, 3))
    for result in results:
        # x1, y1, w, h, n_x, n_y = result
        x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in result]
        # print(x1, y1, w, h, n_x, n_y, scale_factor,img_data.shape,new_img.shape)
        if w == 0 or h == 0:
            continue
        new_img[n_y:n_y + h * scale_factor, n_x:n_x + w * scale_factor, :] = cv2.resize(
            img_data[y1:y1 + h, x1:x1 + w, :], (w * scale_factor, h * scale_factor))

    cv2.imwrite('/home/huangyecheng/dataset/COCO/images/strip_UAVtrain_scale_v1/' + img_name, new_img)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def main():
    device = 'cuda'
    first_detecor_config = '/home/huangyecheng/UAV/mmdetection/configs/DA_DETECTION/fsaf.py'
    first_detecor_config_ckpt = '/home/huangyecheng/UAV/mmdetection/work_dirs/fsaf/epoch_50.pth'
    first_detector = init_detector(first_detecor_config, first_detecor_config_ckpt, device=device)
    train_info = '/home/huangyecheng/dataset/COCO/annotations/instances_UAVtrain_v1.json'
    root = '/home/huangyecheng/dataset/COCO/images/UAVtrain'
    with open(train_info) as f:
        json_info = json.load(f)
    annotation_set = {}
    for annotation in json_info['annotations']:
        image_id = annotation['image_id']
        if not image_id in annotation_set.keys():
            annotation_set[image_id] = []
        annotation_set[image_id].append(annotation)

    coco = COCO(train_info)  # 导入验证集
    size = len(list(coco.imgs.keys()))
    cnt = 1
    anno_id = 1
    new_annotations = []
    new_image = []
    recall_list = []
    area_list = []
    for key in range(size):
        print(cnt, size, end='\r')
        cnt += 1
        img_name = coco.imgs[key]['file_name']
        width = coco.imgs[key]['width']
        height = coco.imgs[key]['height']
        img = os.path.join(root, img_name)
        data = dict(img=img)
        first_results = my_inference_detector(first_detector, LoadImage()(data))
        result = np.concatenate(first_results)
        rec, w, h = UnifiedForegroundPacking(result[:, :4], 1.5, image_shape=[width, height])
        cur_annotation = annotation_set[key]
        flag = [True] * len(cur_annotation)
        image_json = {
            'file_name': img_name,
            'height': h,
            'width': w,
            'id': key
        }
        area_list.append(w * h)
        new_image.append(image_json)
        _sum = 0
        display_merge_result(rec, img, img_name, w, h)
        for bbox in rec:
            x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in bbox]
            t_bbox = [x1, y1, x1 + w, y1 + h]
            for idx, anno in enumerate(cur_annotation):

                [x, y, w, h] = anno['bbox']
                g_bbox = [x, y, x + w, y + h]
                if (compute_iof(t_bbox, g_bbox) > 0.9):
                    new_anno = copy.deepcopy(anno)
                    new_x = max(0, n_x + (x - x1) * scale_factor)
                    new_y = max(0, n_y + (y - y1) * scale_factor)
                    if flag[idx]:
                        _sum += 1
                        flag[idx] = False
                    new_anno['bbox'] = [new_x, new_y, w * scale_factor, h * scale_factor]
                    new_anno['id'] = anno_id
                    new_anno['area'] = w * scale_factor * h * scale_factor
                    anno_id += 1
                    new_annotations.append(new_anno)
        recall_list.append(_sum / len(cur_annotation))
    json_info['images'] = new_image
    json_info['annotations'] = new_annotations
    print(np.mean(recall_list))
    print(np.mean(area_list))
    with open('/home/huangyecheng/dataset/COCO/annotations/instance_merge_UAVtrain_scale_v1.json', 'w') as f:
        json.dump(json_info, f, cls=MyEncoder)


if __name__ == '__main__':
    main()
