import os
import json
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='build UFP images')
    parser.add_argument('visdrone_root', help='the dir to save logs and models')
    parser.add_argument('visdrone_anno', help='the dir to save logs and models')
    parser.add_argument('output_anno', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def main():
    args = parse_args.visdrone_root

    image_root = args.visdrone_root
    anno_root = args.visdrone_anno
    coco = {}

    coco['type'] = 'instances'

    coco['categories'] = [{'supercategory': 'none', 'id': 0, 'name': 'pedestrian'},
                                {'supercategory': 'none', 'id': 1, 'name': 'people'},
                                {'supercategory': 'none', 'id': 2, 'name': 'bicycle'},
                                {'supercategory': 'none', 'id': 3, 'name': 'car'},
                                {'supercategory': 'none', 'id': 4, 'name': 'van'},
                                {'supercategory': 'none', 'id': 5, 'name': 'truck'},
                                {'supercategory': 'none', 'id': 6, 'name': 'tricycle'},
                                {'supercategory': 'none', 'id': 7, 'name': 'awning-tricycle'},
                                {'supercategory': 'none', 'id': 8, 'name': 'bus'},
                                {'supercategory': 'none', 'id': 9, 'name': 'motor'}]
    id_to_name = {}

    id_to_name = {}
    id_to_result = {}
    name_to_id = {}
    images = []
    for idx,filename in enumerate(os.listdir(image_root)):
        img_data = cv2.imread(os.path.join(image_root,filename))
        w,h,c = img_data.shape
        # print(img_data.shape)
        image = {
            'file_name':filename,
            'height':h,
            'width':w,
            'id': idx,
        }
        images.append(image)

    coco['images'] = images
    ids = []
    for image in images:
        filename = image['file_name'].split('.')[0] + '.txt'
        _id = image['id']
        ids.append(_id)
        id_to_name[_id] = filename
        name_to_id[filename] = _id


    new_annotations = []
    anno_id = 0
    for key_id in ids:
        filename = os.path.join(anno_root, id_to_name[key_id])
        anno = open(filename)
        recs = anno.readlines()
        image_id = key_id
        for rec in recs:
            rec = rec.strip()
            xmin, ymin, o_width, o_height, s, _cls = [int(_) for _ in rec.split(',')[:6]]
            xmax = xmin + o_width
            ymax = ymin + o_height
            if _cls == 0 or _cls == 11:
                continue
            annotation = dict()
            annotation['area'] = o_width * o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = _cls - 1
            annotation['id'] = anno_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            new_annotations.append(annotation)
            anno_id += 1

    coco['annotations'] = new_annotations
    with open(args.output_anno, 'w') as f:
        json.dump(coco, f)

if __name__ == '__main__':
    main()