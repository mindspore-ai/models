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

"""Evaluation for nasfpn"""

import os
import argparse
import numpy as np
from PIL import Image
from src.coco_eval import metrics

parser = argparse.ArgumentParser(description='nasfpn postprocess')
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--dataset_path", type=str, required=True, help="dataset path.")
parser.add_argument("--anno_path", type=str, required=True, help="annotation json path.")
args = parser.parse_args()

def get_pred(result_path, img_id):
    """get prediction output"""
    boxes_file = os.path.join(result_path, img_id + '_0.bin')
    scores_file = os.path.join(result_path, img_id + '_1.bin')

    boxes = np.fromfile(boxes_file, dtype=np.float32).reshape(76725, 4)
    scores = np.fromfile(scores_file, dtype=np.float32).reshape(76725, 81)
    return boxes, scores

def get_img_size(file_name):
    """get image size"""
    img = Image.open(file_name)
    return img.size

def get_img_set(anno_json_path):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO
    need_img_ids = []
    coco = COCO(anno_json_path)
    image_ids = coco.getImgIds()
    print("first dataset is {}".format(len(image_ids)))
    for img_id in image_ids:
        iscrowd = False
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        for label in anno:
            iscrowd = iscrowd or label["iscrowd"]
        if iscrowd:
            continue
        need_img_ids.append(img_id)
    return need_img_ids

def cal_acc(result_path, img_path, anno_path):
    """calculate accuracy"""
    need_img_ids = get_img_set(anno_path)

    imgs = os.listdir(img_path)
    pred_data = []

    for img in imgs:
        img_id = img.split('.')[0]
        if int(img_id) not in need_img_ids:
            continue
        boxes, box_scores = get_pred(result_path, img_id)

        w, h = get_img_size(os.path.join(img_path, img))
        img_shape = np.array((h, w), dtype=np.float32)
        pred_data.append({"boxes": boxes,
                          "box_scores": box_scores,
                          "img_id": int(img_id),
                          "image_shape": img_shape})

    mAP = metrics(pred_data, anno_path)
    print(f"mAP: {mAP}")

if __name__ == '__main__':
    cal_acc(args.result_path, args.dataset_path, args.anno_path)
