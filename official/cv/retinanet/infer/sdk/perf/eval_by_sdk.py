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
"""eval for coco map"""
import argparse
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='"Retinanet eval " "example."')

# Gets the full path to the pro script
current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

parser.add_argument(
    "--res_path",
    type=str,
    help="Get the JSON directory of inferred results",
    default=os.path.join(current_path, "result/predictions_test.json"),
    required=False,
)
parser.add_argument(
    "--instances_path",
    type=str,
    help="The annotation file directory for the COCO dataset",
    default=os.path.join(current_path, "dataset/annotations/instances_val2017.json"),
)

def get_eval_result():
    # Filter the blue samples in the script
    coco_gt = COCO(args.instances_path)
    image_id_flag = coco_gt.getImgIds()
    need_img_ids = []
    for img_id in image_id_flag:
        iscrowd = False
        anno_ids = coco_gt.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco_gt.loadAnns(anno_ids)
        for label in anno:
            iscrowd = iscrowd or label["iscrowd"]
        if iscrowd:
            continue
        need_img_ids.append(img_id)

    # Get the eval value
    coco_dt = coco_gt.loadRes(args.res_path)
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = sorted(need_img_ids)
    E.evaluate()
    E.accumulate()
    E.summarize()
    mAP = E.stats[0]
    print("\n========================================\n")
    print(f"mAP: {mAP}")


if __name__ == '__main__':
    args = parser.parse_args()
    get_eval_result()
