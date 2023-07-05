# Copyright 2023 Huawei Technologies Co., Ltd
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
"""coco eval"""
import contextlib
import copy
import datetime
import json
import os
import sys

import numpy as np
from mindspore import Tensor, ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.focus_detr import box_ops


def merge(img_ids, eval_imgs):
    all_img_ids = [img_ids]
    all_eval_imgs = [eval_imgs]

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())
    ##
    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


class Redirct:
    """Redirct"""

    def __init__(self):
        self.content = ""

    def write(self, content):
        """write"""
        self.content += content

    def flush(self):
        """flush"""
        self.content = ""


#
class CocoEvaluator:
    """coco evaluator"""

    def __init__(self, coco_gt, iou_types, useCats=True):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        #
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats

    def old_update(self, predictions):
        """update"""
        results = self.prepare_for_coco_detection(predictions)
        self.results.extend(results)

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            # results = self.prepare(predictions, iou_type)
            results = self.prepare_for_coco_detection(predictions)
            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def write_result(self):
        """Save result to file."""
        t = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        if self.save_prefix:
            self.file_path = self.save_prefix + "/predict" + t + ".json"
        else:
            self.file_path = "predict" + t + ".json"
        with open(self.file_path, "w") as f:
            json.dump(self.results, f)

    def prepare(self, predictions, iou_type):
        return self.prepare_for_coco_detection(predictions)

    def get_eval_result(self):
        """Get eval result."""
        coco_gt = self.coco_gt
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        sys.stdout = stdout
        return rdct.content

    def old_prepare_for_coco_detection(self, predictions):
        """prepare for coco detection"""
        coco_results = []
        for original_id, prediction in predictions.items():
            if not prediction:
                continue

            boxes = prediction["boxes"]
            boxes = box_ops.box_xyxy_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": int(original_id),
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if not prediction:
                continue
            boxes = prediction["boxes"]
            boxes = boxes.asnumpy()
            boxes = box_ops.box_xyxy_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].asnumpy().tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].asnumpy().tolist()
            else:
                labels = prediction["labels"]
            try:
                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            except Exception as e:
                print(f"Error '{e.message}' occurred. Arguments {e.args}.")
        return coco_results

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])


def old_post_process(outputs, target_sizes):
    """Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"].asnumpy()
    ##
    # pdb.set_trace()
    prob = ops.Softmax(axis=-1)(out_logits)  # 用Softmax
    labels, scores = ops.ArgMaxWithValue(axis=-1)(prob[..., :-1])
    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = np.array_split(target_sizes, 2, axis=1)
    scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=2).squeeze()
    boxes = boxes * scale_fct.reshape(-1, 1, 4)

    results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores.asnumpy(), labels.asnumpy(), boxes)]
    print(f"---results:{results}")
    return results


def post_process(outputs, target_sizes):
    """Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
    sigmoid = ops.Sigmoid()

    prob = sigmoid(out_logits)
    new_prob = prob.view(out_logits.shape[0], -1)

    num_select = 300
    topk_values, topk_indexes = ops.TopK(sorted=True)(new_prob, num_select)
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    out_bbox = out_bbox.asnumpy()
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = Tensor(boxes)
    boxes = ops.gather(boxes, topk_boxes[0], 1)

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = np.array_split(target_sizes, 2, axis=1)
    scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=2).squeeze().reshape(-1, 1, 4)
    scale_fct = Tensor(scale_fct)
    boxes = boxes * scale_fct

    results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(topk_values, labels, boxes)]
    return results


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {(imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds for areaRng in p.areaRng for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs
