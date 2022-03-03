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

import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..decorator import process_cfg


def apply_threshold(in_file, threshold):
    """
    apply threshold
    """
    out_file = in_file[:-5] + '-' + str(threshold) + '.json'

    with open(in_file) as data_file:
        data = json.load(data_file)

    for person_id in range(len(data)):
        keypoints = data[person_id]["keypoints"]
        keypoints = [int(keypoints[i] > threshold) if i % 3 == 2 else int(keypoints[i]) for i in range(len(keypoints))]
        data[person_id]["keypoints"] = keypoints

    with open(out_file, 'w') as outfile:
        json.dump(data, outfile)

    return out_file


def eval_init(cfg, prediction=None):
    """
    init
    """
    dataset_path = cfg.dataset.path
    dataset_phase = cfg.dataset.phase
    dataset_ann = cfg.dataset.ann
    threshold = 0

    # initialize coco_gt api
    ann_file = '%s/annotations/%s_%s.json' % (dataset_path, dataset_ann, dataset_phase)
    coco_gt = COCO(ann_file)

    # initialize coco_pred api
    pred_file = apply_threshold(prediction or cfg.gt_segm_output, threshold)
    coco_pred = coco_gt.loadRes(pred_file)

    return coco_gt, coco_pred


@process_cfg
def eval_coco(cfg=None, prediction=None):
    """
    eval coco entry
    """
    coco_gt, coco_pred = eval_init(cfg, prediction)
    eval_mscoco_with_segm(coco_gt, coco_pred)


def eval_mscoco_with_segm(coco_gt, coco_pred):
    """
    eval mscoco

    Args:
        coco_gt: ground truth
        coco_pred: prediction
    """
    # running evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
