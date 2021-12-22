import math
import random
import numpy as np
from .spp import phsppog

def scale_boxes(bboxes, scale, image_shape=[1333, 1333]):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.shape[1] == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * 0.5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * 0.5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * 0.5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale
    w, h = image_shape

    boxes_scaled = np.zeros_like(bboxes)
    boxes_scaled[:, 0] = np.clip(x_c - w_half, 0, w - 1)
    boxes_scaled[:, 2] = np.clip(x_c + w_half, 0, w - 1)
    boxes_scaled[:, 1] = np.clip(y_c - h_half, 0, h - 1)
    boxes_scaled[:, 3] = np.clip(y_c + h_half, 0, h - 1)
    return boxes_scaled


def get_merge_bbox_aera(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x21, y21, x22, y22 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    bbox1_area = (x12 - x11) * (y12 - y11)
    bbox2_area = (x22 - x21) * (y22 - y21)
    x_min = min(x11, x21)
    y_min = min(y11, y21)
    x_max = max(x12, x22)
    y_max = max(y12, y22)
    merge_area = (x_max - x_min) * (y_max - y_min)
    return merge_area, bbox1_area + bbox2_area, [x_min, y_min, x_max, y_max]


def get_bbox_area(bbox1):
    x11, y11, x12, y12 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    bbox1_area = (x12 - x11) * (y12 - y11)
    return bbox1_area



def ForegroundRegionGeneration(bbox_list, scaled_bbox_list):
    num_bbox = bbox_list.shape[0]
    x1 = bbox_list[:, 0]
    y1 = bbox_list[:, 1]
    x2 = bbox_list[:, 2]
    y2 = bbox_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    avg_areas = areas
    cnt = np.array([1] * num_bbox)
    is_used = [True] * num_bbox
    for idx in range(num_bbox):
        if not is_used[idx]:
            continue
        A = scaled_bbox_list[idx]
        for jdx in range(num_bbox):
            if not is_used[jdx] or idx == jdx:
                continue
            merge_area, origin_area, merge_bbox = get_merge_bbox_aera(A, scaled_bbox_list[jdx])
            if merge_area < origin_area:
                A = merge_bbox
                is_used[jdx] = False
                avg_areas[idx] += avg_areas[jdx]
                cnt[idx] += cnt[jdx]
        scaled_bbox_list[idx] = A
    avg_areas = avg_areas/cnt
    scale_factor = np.array([1] * num_bbox)
    for idx in range(num_bbox):
        if avg_areas[idx] < 32 * 32:
            scale_factor[idx] = 4
        elif avg_areas[idx] < 96 * 96:
            scale_factor[idx] = 2
        else:
            scale_factor[idx] = 1
    return scaled_bbox_list[is_used], scale_factor[is_used]


def ForegroundRegionScaleEqualization(bbox_list, foreground_region):
    x1 = bbox_list[:, 0]
    y1 = bbox_list[:, 1]
    x2 = bbox_list[:, 2]
    y2 = bbox_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    avg_areas = [0] * foreground_region.shape[0]
    for idx in range(len(foreground_region)):
        xx1 = np.maximum(foreground_region[idx,0], x1)
        yy1 = np.maximum(foreground_region[idx,1], y1)
        xx2 = np.minimum(foreground_region[idx,2], x2)
        yy2 = np.minimum(foreground_region[idx,3], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / areas

        inds = np.where(ovr > 0.95)[0]
        avg_area = areas[inds].sum()/(inds.sum())
        avg_areas[idx] = avg_area

    scale_factor = [1] * foreground_region.shape[0]
    for idx in range(foreground_region.shape[0]):
        # scale_factor[idx] = math.sqrt(96 * 96 / avg_areas[idx])
        if avg_areas[idx] < 32 * 32:
            scale_factor[idx] = 4
        elif avg_areas[idx] < 96 * 96:
            scale_factor[idx] = 2
        else:
            scale_factor[idx] = 1
    

    return scale_factor
    
def Packing(foreground_region, scale_factor, output_shape=[1333,800]):
    boxes = []
    for idx, _flag in enumerate(scale_factor):
        w = foreground_region[idx][2] - foreground_region[idx][0]
        h = foreground_region[idx][3] - foreground_region[idx][1]
        factor = scale_factor[idx]

        boxes.append([w * factor, h * factor])
    width_min = 300
    width_max = 2666
    while (width_min <= width_max):
        width_mid = (width_min + width_max) / 2
        height, rectangles = phsppog(width_mid, boxes, sorting='height')
        if height > width_mid:
            width_min = width_mid + 1
        else:
            width_max = width_mid - 1

    flag = [True] * foreground_region.shape[0]
    result = []
    new_width = 0
    new_height = 0
    for post_rec in rectangles:
        x = post_rec.x
        y = post_rec.y
        w = post_rec.w
        h = post_rec.h
        new_width = max(new_width, x + w)
        new_height = max(new_height, y + h)
        for idx in range(foreground_region.shape[0]):
            if not flag[idx]:
                continue
            factor = scale_factor[idx]
            _w = foreground_region[idx, 2] - foreground_region[idx, 0]
            _h = foreground_region[idx, 3] - foreground_region[idx, 1]
            if _w * factor == w and _h * factor == h:
                flag[idx] = False
                result.append([foreground_region[idx, 0], foreground_region[idx, 1], _w, _h, x, y, factor])

    return result, new_width, new_height
    


def UnifiedForegroundPacking(bbox_list, scale, input_shape, output_shape=[1333,800]):
    
    # scale bbox
    scaled_bbox_list = scale_boxes(bbox_list, scale, input_shape)
    # Foreground Region Generation Algorithm
    foreground_region, scale_factor = ForegroundRegionGeneration(bbox_list, scaled_bbox_list)


    # Foreground Region Scale Equalization
    # scale_factor = ForegroundRegionScaleEqualization(bbox_list, foreground_region)

    # Packing
    result, new_width, new_height = Packing(foreground_region, scale_factor, output_shape)

    return result, new_width, new_height 


if __name__ == '__main__':
    boxes = [
        [5, 3,10,10], [5, 3,10,10], [2, 4,10,10], [30, 8,10,10], [10, 20,10,10],
        [20, 10,10,10], [5, 5,10,10], [5, 5,10,10], [10, 10,10,10], [10, 5,10,10],
        [6, 4,10,10], [1, 10,10,10], [8, 4,10,10], [6, 6,10,10], [20, 14,10,1000]
    ]
    new_out = UnifiedForegroundPacking(np.array(boxes), 1.5, (1333,1333))
    print(new_out)


    # print(out)
    # boxes = [
    #     [5, 3], [5, 3], [2, 4], [30, 8], [10, 20],
    #     [20, 10], [5, 5], [5, 5], [10, 10], [10, 5],
    #     [6, 4], [1, 10], [8, 4], [6, 6], [20, 14]
    # ]
    # print(len(boxes))
    # width = 10
    # height, rectangles = phsppog(width, boxes)
    # print(height)
    # print(rectangles)
    # print("The height is: {}".format(height))
