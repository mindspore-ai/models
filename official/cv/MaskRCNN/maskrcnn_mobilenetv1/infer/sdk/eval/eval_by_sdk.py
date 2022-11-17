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


"""post process for 310 inference"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from config import config as cfg
from eval.util import coco_eval, results2json


def get_img_size(file_name):
    img = Image.open(file_name)
    return img.size


def parse_result(result_file, img_size, num_classes):
    all_box = [[] for i in range(0, num_classes)]
    cls_segms = [[] for _ in range(num_classes)]
    if not os.path.exists(result_file):
        print(f"No such file({result_file}), will be ignore.")
        return [np.asarray(box) for box in all_box], cls_segms

    with open(result_file, 'r') as fp:
        result = json.loads(fp.read())

    data = result.get("MxpiObject")
    if not data:
        return [np.asarray(box) for box in all_box], cls_segms

    for bbox in data:
        im_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        class_vec = bbox.get("classVec")[0]
        np_bbox = np.array([
            float(bbox["x0"]),
            float(bbox["y0"]),
            float(bbox["x1"]),
            float(bbox["y1"]),
            class_vec.get("confidence")
        ])
        all_box[int(class_vec["classId"])].append(np_bbox)
        if "imageMask" not in bbox:
            print(f"bbox: {bbox}, result file: {result_file}")

        mask_data = bbox["imageMask"]["data"]
        mask_width = bbox["imageMask"]["shape"][1]
        mask_height = bbox["imageMask"]["shape"][0]
        if len(mask_data) != mask_width * mask_height:
            print("The mask result data is error.")
            return [np.asarray(box) for box in all_box], cls_segms

        mask = [255 if i == '1' else 0 for i in mask_data]
        mask = np.array(mask)
        mask = mask.reshape(mask_height, mask_width)
        x, y = int(np_bbox[0]), int(np_bbox[1])

        im_mask[y:y + mask_height, x:x + mask_width] = mask

        rle = mask_utils.encode(np.array(im_mask[:, :, np.newaxis],
                                         order='F'))[0]
        cls_segms[int(class_vec["classId"])].append(rle)

    return [np.asarray(box) for box in all_box], cls_segms


def get_eval_result(ann_file, img_path, result_paths):
    outputs = []

    dataset_coco = COCO(ann_file)
    img_ids = dataset_coco.getImgIds()

    for img_id in img_ids:
        file_id = str(img_id).zfill(12)
        file_path = os.path.join(img_path, f"{file_id}.jpg")
        img_size = get_img_size(file_path)
        result_json = os.path.join(result_paths, f"{file_id}.json")
        bbox_results, segm_results = parse_result(result_json, img_size,
                                                  cfg.NUM_CLASSES)
        outputs.append((bbox_results, segm_results))

    eval_types = ["bbox", "segm"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    coco_eval(result_files, eval_types, dataset_coco, single_result=False)


if __name__ == '__main__':
    result_path = "./result"
    parser = argparse.ArgumentParser(description="maskrcnn inference")
    parser.add_argument("--ann_file",
                        type=str,
                        required=True,
                        help="ann file.")
    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image file path.")
    args = parser.parse_args()
    get_eval_result(args.ann_file, args.img_path, result_path)
