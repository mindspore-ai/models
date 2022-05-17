# Copyright 2022 Huawei Technologies Co., Ltd
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
import json
import numpy as np
import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_classes = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie",
                "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

parser = argparse.ArgumentParser(description='nasfpn postprocess')
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--dataset_path", type=str, required=True, help="dataset path.")
parser.add_argument("--anno_path", type=str, required=True, help="annotation json path.")
args = parser.parse_args()

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def get_pred(result_path, img_id):
    """get prediction output"""
    boxes_file = os.path.join(result_path, img_id + '_0.bin')
    scores_file = os.path.join(result_path, img_id + '_1.bin')

    boxes = np.fromfile(boxes_file, dtype=np.float32).reshape(76725, 4)
    scores = np.fromfile(scores_file, dtype=np.float32).reshape(76725, 81)
    scores = sigmoid(scores)
    return boxes, scores

def get_img_size(file_name):
    """get image size"""
    img = Image.open(file_name)
    return img.size

def get_img_set(anno_json_path):
    """Get image path and annotation from COCO."""
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

def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep

def cal_acc(result_path, img_path, anno_path):
    """calculate accuracy"""
    need_img_ids = get_img_set(anno_path)

    imgs = sorted(os.listdir(img_path))
    predictions = []
    img_ids = []

    val_cls = coco_classes
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls
    coco_gt = COCO(anno_path)
    classs_dict = {}
    cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["name"]] = cat["id"]

    for img in tqdm.tqdm(imgs[:]):
        img_id = img.split('.')[0]
        if int(img_id) not in need_img_ids:
            continue
        pred_boxes, box_scores = get_pred(result_path, img_id)

        w, h = get_img_size(os.path.join(img_path, img))
        img_id = int(img_id)
        final_boxes = []
        final_label = []
        final_score = []
        img_ids.append(img_id)

        for c in range(1, 81):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > 0.1
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, 0.6, 100)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
            res['score'] = score
            res['category_id'] = label
            predictions.append(res)

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)
    del predictions
    coco_dt = coco_gt.loadRes('predictions.json')
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    mAP = E.stats[0]
    print("\n========================================\n")
    print(f"mAP: {mAP}")

if __name__ == '__main__':
    cal_acc(args.result_path, args.dataset_path, args.anno_path)
