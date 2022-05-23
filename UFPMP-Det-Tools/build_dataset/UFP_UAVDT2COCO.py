import argparse
from mmdet.apis import init_detector, show_result_pyplot, inference_detector
import warnings
import cv2
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
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


def ufp_inference_detector(model, data):
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


def draw_ufp_result(results, img, img_name, w, h, anno_root=None):
    w = math.ceil(w)
    h = math.ceil(h)
    img_data = cv2.imread(img)
    if anno_root:
        anno_path = os.path.join(anno_root, img_name[:-4] + '.txt')
        anno = open(anno_path)
        for rec in anno.readlines():
            rec = rec.strip()
            x, y, _w, _h, s, _cls = [int(_) for _ in rec.split(',')[:6]]
            if _cls == 0:
                img_data[y:y + _h, x:x + _w, :] = 0
        anno.close()
    new_img = np.zeros((h, w, 3))
    for result in results:
        x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in result]
        if w == 0 or h == 0:
            continue
        new_img[n_y:n_y + h * scale_factor, n_x:n_x + w * scale_factor, :] = cv2.resize(
            img_data[y1:y1 + h, x1:x1 + w, :], (w * scale_factor, h * scale_factor))
    
    return new_img

    # cv2.imwrite('/home/huangyecheng/dataset/COCO/images/strip_UAVtrain_scale_v1/' + img_name, new_img)


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

def parse_args():
    parser = argparse.ArgumentParser(description='build UFP images')
    parser.add_argument('detector_config', help='the dir to save logs and models')
    parser.add_argument('detector_ckpt', help='the dir to save logs and models')
    parser.add_argument('dataset_root', help='the dir to save logs and models')
    parser.add_argument('dataset_anno', help='the dir to save logs and models')
    parser.add_argument('ufp_images_output_dir', help='the dir to save logs and models')
    parser.add_argument('ufp_anno_output_path', help='the dir to save logs and models')
    parser.add_argument('--txt_path', help='the dir to save logs and models', default=None)
    args = parser.parse_args()
    return args

def main():
    device = 'cuda'
    args = parse_args()
    detecor_config = args.detector_config
    detecor_config_ckpt = args.detector_ckpt
    dataset_root = args.dataset_root
    dataset_anno = args.dataset_anno
    ufp_images_output_dir = args.ufp_images_output_dir
    ufp_anno_output_path = args.ufp_anno_output_path
    txt_path = args.txt_path


    detector = init_detector(detecor_config, detecor_config_ckpt, device=device)
    with open(dataset_anno) as f:
        json_info = json.load(f)
    annotation_set = {}
    for annotation in json_info['annotations']:
        image_id = annotation['image_id']
        if not image_id in annotation_set.keys():
            annotation_set[image_id] = []
        annotation_set[image_id].append(annotation)

    coco = COCO(dataset_anno)  # 导入验证集
    size = len(list(coco.imgs.keys()))
    cnt = 1
    anno_id = 1
    new_annotations = []
    new_image = []
    for key in range(size):
        print(cnt, size, end='\r')
        cnt += 1
        img_name = coco.imgs[key]['file_name']
        width = coco.imgs[key]['width']
        height = coco.imgs[key]['height']
        img = os.path.join(dataset_root, img_name)
        # data = dict(img=img)
        first_results = inference_detector(detector, img)
        result = np.concatenate(first_results)
        rec, w, h = UnifiedForegroundPacking(result[:, :4], 1.5, input_shape=[width, height])
        cur_annotation = annotation_set[key]
        flag = [True] * len(cur_annotation)
        image_json = {
            'file_name': img_name,
            'height': h,
            'width': w,
            'id': key
        }
        new_image.append(image_json)
        ufp_img = draw_ufp_result(rec, img, img_name, w, h, txt_path)
        cv2.imwrite(os.path.join(ufp_images_output_dir, img_name), ufp_img)

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
                        flag[idx] = False
                    new_anno['bbox'] = [new_x, new_y, w * scale_factor, h * scale_factor]
                    new_anno['id'] = anno_id
                    new_anno['area'] = w * scale_factor * h * scale_factor
                    anno_id += 1
                    new_annotations.append(new_anno)
    json_info['images'] = new_image
    json_info['annotations'] = new_annotations
    with open(ufp_anno_output_path, 'w') as f:
        json.dump(json_info, f, cls=MyEncoder)


if __name__ == '__main__':
    main()
