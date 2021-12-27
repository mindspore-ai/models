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
"""
Accuracy calculation of model
"""

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(description='CenterNet evaluation')
parser.add_argument(
    "-g",
    type=str,
    default="",
    help="gt_anno")
parser.add_argument(
    "-p",
    type=str,
    default="",
    help="pred_anno")
args_opt = parser.parse_args()

def run_eval(gt_anno, pred_anno):
    """evaluation by coco api"""
    coco = COCO(gt_anno)
    coco_dets = coco.loadRes(pred_anno)
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
if __name__ == '__main__':
    run_eval(args_opt.g, args_opt.p)
