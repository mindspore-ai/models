
# Copyright(C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Run test"""

import json
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

classes_path = sys.argv[1]
imgid_path = sys.argv[2]
prediction_path = sys.argv[3]
coco_path = sys.argv[4]

print(classes_path, imgid_path, prediction_path, coco_path)

with open(classes_path, encoding='utf-8') as f:
    val_cls = json.load(f)

val_cls_dict = {}
for i, cls in enumerate(val_cls):
    val_cls_dict[i] = cls

coco_gt = COCO(coco_path)
classs_dict = {}
cat_ids = coco_gt.loadCats(coco_gt.getCatIds())


catIds = []
for cat in cat_ids:
    if cat["name"] in val_cls:
        catIds.append(cat["id"])
    classs_dict[cat["name"]] = cat["id"]

new_idx = []
for v in val_cls:
    if classs_dict.get(v):
        new_idx.append(classs_dict[v])
    else:
        new_idx.append(0)

with open(imgid_path, encoding='utf-8') as f:
    img_ids = json.load(f)

coco_dt = coco_gt.loadRes(prediction_path)

E = COCOeval(coco_gt, coco_dt, iouType='bbox')
E.params.imgIds = img_ids
E.params.catIds = catIds
E.evaluate()
E.accumulate()
E.summarize()

print('mAP:', E.stats[0])
