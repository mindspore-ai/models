# !/usr/bin/env python

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_eval_result(ann_file, result_file):
    """Get eval result."""
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="yolov5 eval")
    parser.add_argument('--ann_file', type=str, default='', help='path to annotation')
    parser.add_argument('--result_file', type=str, default='', help='path to annotation')

    args = parser.parse_args()

    get_eval_result(args.ann_file, args.result_file)
