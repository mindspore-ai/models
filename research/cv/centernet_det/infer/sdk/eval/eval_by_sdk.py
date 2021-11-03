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
"""accuracy calculation"""
import os
import json
import argparse
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval


def get_eval_result(ann_file, result_path):
    '''get evaluation results'''
    outputs = []
    coco_anno = coco.COCO(ann_file)
    img_ids = coco_anno.getImgIds()
    for img_id in img_ids:
        file_id = str(img_id).zfill(12)
        result_json = os.path.join(result_path, f"infer_{file_id}_result.json")
        with open(result_json, 'r') as fp:
            ann = json.loads(fp.read())
        for i in range(len(ann)):
            outputs.append(ann[i])

    return outputs


def cal_acc(ann_file, result_path):
    '''calculate inference accuracy'''
    outputs = get_eval_result(ann_file, result_path)
    coco_anno = coco.COCO(ann_file)
    coco_dets = coco_anno.loadRes(outputs)
    coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="centernet inference")
    parser.add_argument("--ann_file",
                        type=str,
                        required=True,
                        help="ann file.")
    parser.add_argument("--result_path",
                        type=str,
                        required=True,
                        help="inference result save path.")
    args = parser.parse_args()
    cal_acc(args.ann_file, args.result_path)
