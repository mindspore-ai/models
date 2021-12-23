# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""post process for 310 inference"""
import os
import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from src.model_utils.config import config

dst_width = config.img_width
dst_height = config.img_height


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size


def get_resize_ratio(img_size):
    org_width, org_height = img_size
    resize_ratio = dst_width / org_width
    if resize_ratio > dst_height / org_height:
        resize_ratio = dst_height / org_height

    return resize_ratio


def compute_area(point):
    s = 0.0
    point_num = len(point)
    if point_num < 3:
        return s
    for i in range(point_num):
        s += point[i][1] * (point[i-1][0]-point[(i+1)%point_num][0])
    return abs(s/2.0)


def save_result(masks, boxes, labels, img_metas_, txt):
    """save the detection result"""
    f = open(txt, 'w')
    for k in range(len(masks)):
        box = boxes[k].tolist()
        label = labels[k].tolist()
        if label == 0 and box[-1] > 0.9:
            scale = [img_metas_[3], img_metas_[2], img_metas_[3], img_metas_[2]]
            [x1, y1, x2, y2] = [int(box[l] / scale[l]) for l in range(len(box) - 1)]
            w, h = x2 - x1, y2 - y1
            image_height, image_width = int(img_metas_[0]), int(img_metas_[1])
            if x2 > image_width or y2 > image_height or w <= 0 or h <= 0:
                continue

            mask = masks[k].tolist()
            mask = np.array(mask)
            mask = mask > 0.5
            mask = mask.astype(np.uint8)
            mask = mask * 255
            mask = cv2.resize(mask, (w, h))
            canvas = np.zeros((image_height, image_width))
            canvas[y1:y2, x1:x2] = mask
            temp = cv2.findContours(canvas.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            temp = temp[-2]
            temp = [x.flatten() for x in temp]
            temp = [x for x in temp if len(x) > 6]
            poly = temp
            poly = poly[0]
            point = []
            for i in range(0, len(poly) - 1, 2):
                point.append([poly[i], poly[i + 1]])
            area = compute_area(point)
            if area < 120:
                continue
            for p in range(0, len(poly) - 1):
                f.write(str(poly[p]) + ',')
            f.write(str(poly[len(poly) - 1]) + '\n')
    f.close()


def get_eval_result(ann_file, img_path, result_path):
    """ Get metrics result according to the annotation file and result file"""
    max_num = 128
    result_path = result_path

    dataset_coco = COCO(ann_file)
    img_ids = dataset_coco.getImgIds()
    imgs = dataset_coco.imgs
    if not os.path.exists('temp'):
        os.mkdir('temp')
    for img_id in img_ids:
        file = img_path + imgs[img_id]['file_name']
        img_size = get_img_size(file)
        resize_ratio = get_resize_ratio(img_size)

        img_name = imgs[img_id]['file_name']

        img_metas = np.array([img_size[1], img_size[0]] + [resize_ratio, resize_ratio])

        bbox_result_file = os.path.join(result_path, img_name.split('.')[0] + "_0.bin")
        label_result_file = os.path.join(result_path, img_name.split('.')[0] + "_1.bin")
        mask_result_file = os.path.join(result_path, img_name.split('.')[0] + "_2.bin")
        mask_fb_result_file = os.path.join(result_path, img_name.split('.')[0] + "_3.bin")

        all_bbox = np.fromfile(bbox_result_file, dtype=np.float16).reshape(1, 100, 5)
        all_label = np.fromfile(label_result_file, dtype=np.int32).reshape(1, 100, 1)
        all_mask = np.fromfile(mask_result_file, dtype=np.bool_).reshape(1, 100, 1)
        all_mask_fb = np.fromfile(mask_fb_result_file, dtype=np.float16).reshape(1, 100, 64, 28, 28)

        all_bbox_squee = np.squeeze(all_bbox)
        all_label_squee = np.squeeze(all_label)
        all_mask_squee = np.squeeze(all_mask)
        all_mask_fb_squee = np.squeeze(all_mask_fb)

        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]
        _all_mask_fb_tmp_mask = all_mask_fb_squee[all_mask_squee, :, :, :]
        all_mask_fb_tmp_mask = np.zeros((all_bboxes_tmp_mask.shape[0], 28, 28))
        for i in range(all_bboxes_tmp_mask.shape[0]):
            all_mask_fb_tmp_mask[i, :, :] = _all_mask_fb_tmp_mask[i, all_labels_tmp_mask[i]+1, :, :]
        if all_bboxes_tmp_mask.shape[0] > max_num:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
            inds = inds[:max_num]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]
            all_mask_fb_tmp_mask = all_mask_fb_tmp_mask[inds]
        save_result(all_mask_fb_tmp_mask, all_bboxes_tmp_mask, all_labels_tmp_mask,
                    img_metas, 'temp/' + img_name.replace('.jpg', '.txt'))


if __name__ == '__main__':
    get_eval_result(config.ann_file, config.img_path, config.result_path)
    os.system('cd temp && rm -rf temp.zip && zip temp.zip *.txt && cd .. && '
              'python eval_code/curved_tiou/script.py -g=total-text-gt.zip -s=temp/temp.zip')
