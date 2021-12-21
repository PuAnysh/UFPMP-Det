from my_tools.ph import *
import numpy as np


def scale_boxes(bboxes, scale, image_shape=[1333, 1333]):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.shape[1] == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

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



def ForegroundRegionGeneration(bbox_list, scale, image_shape):
    num_bbox = bbox_list.shape[0]
    avg_area = [0] * num_bbox
    num_cnt = [0] * num_bbox


def merge_bbox(bbox_list, scale=None, image_shape=None):
    num_bbox = bbox_list.shape[0]
    avg_area = [0] * num_bbox
    num_cnt = [0] * num_bbox
    if not image_shape:
        image_shape = [1333, 1333]
    for i in range(num_bbox):
        avg_area[i] = get_bbox_area(bbox_list[i])
        num_cnt[i] += 1
    if scale:
        # print(bbox_list)
        bbox_list = scale_boxes(bbox_list, scale, image_shape)

    flag = [True] * num_bbox
    for idx in range(num_bbox):
        if not flag[idx]:
            continue
        for jdx in range(num_bbox):
            if not flag[jdx] or idx == jdx:
                continue
            merge_area, origin_area, merge_bbox = get_merge_bbox_aera(bbox_list[idx], bbox_list[jdx])
            if merge_area < origin_area:
                bbox_list[idx] = merge_bbox
                avg_area[idx] += avg_area[jdx]
                num_cnt[idx] += num_cnt[jdx]
                flag[jdx] = False
    # avg_area = [0] * num_bbox
    scale_factor = [1] * num_bbox
    for idx in range(num_bbox):
        avg_area[idx] = avg_area[idx] / num_cnt[idx]
        if avg_area[idx] < 32 * 32:
            scale_factor[idx] = 4
        elif avg_area[idx] < 96 * 96:
            scale_factor[idx] = 2
        else:
            scale_factor[idx] = 1
    boxes = []
    for idx, _flag in enumerate(flag):
        if _flag:
            w = bbox_list[idx][2] - bbox_list[idx][0]
            h = bbox_list[idx][3] - bbox_list[idx][1]
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
        for idx in range(num_bbox):
            if not flag[idx]:
                continue
            factor = scale_factor[idx]
            _w = bbox_list[idx][2] - bbox_list[idx][0]
            _h = bbox_list[idx][3] - bbox_list[idx][1]
            if _w * factor == w and _h * factor == h:
                flag[idx] = False
                result.append([bbox_list[idx][0], bbox_list[idx][1], _w, _h, x, y, factor])

    return result, new_width, new_height


if __name__ == '__main__':
    boxes = [
        [5, 3], [5, 3], [2, 4], [30, 8], [10, 20],
        [20, 10], [5, 5], [5, 5], [10, 10], [10, 5],
        [6, 4], [1, 10], [8, 4], [6, 6], [20, 14]
    ]
    print(len(boxes))
    width = 10
    height, rectangles = phsppog(width, boxes)
    print(height)
    print(rectangles)
    print("The height is: {}".format(height))
