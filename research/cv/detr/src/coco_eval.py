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
"""coco eval"""
import copy
import datetime
import json
import sys

import numpy as np
from mindspore import ops
from pycocotools.cocoeval import COCOeval

from src import box_ops


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


class CocoEvaluator:
    """coco evaluator"""
    def __init__(self, coco_gt, output_path=''):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.results = []
        self.save_prefix = output_path

    def update(self, predictions):
        """update"""
        results = self.prepare_for_coco_detection(predictions)
        self.results.extend(results)

    def write_result(self):
        """Save result to file."""
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        if self.save_prefix:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
        else:
            self.file_path = 'predict' + t + '.json'
        with open(self.file_path, 'w') as f:
            json.dump(self.results, f)

    def get_eval_result(self):
        """Get eval result."""
        coco_gt = self.coco_gt
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        sys.stdout = stdout
        return rdct.content

    def prepare_for_coco_detection(self, predictions):
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


def post_process(outputs, target_sizes):
    """ Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes'].asnumpy()

    prob = ops.Softmax(axis=-1)(out_logits)
    labels, scores = ops.ArgMaxWithValue(axis=-1)(prob[..., :-1])
    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = np.array_split(target_sizes, 2, axis=1)
    scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=2).squeeze()
    boxes = boxes * scale_fct.reshape(-1, 1, 4)

    results = [{'scores': s, 'labels': l, 'boxes': b}
               for s, l, b in zip(scores.asnumpy(), labels.asnumpy(), boxes)]
    return results
