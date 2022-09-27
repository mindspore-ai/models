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
"""post process for 310 inference"""
import os
import argparse
import numpy as np

from pycocotools.coco import COCO
from src.util import coco_eval, bbox2result_1image, results2json
from src.model_utils.config import config


def prepare_args():
    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument('--anno_path', type=str)
    parser.add_argument("--result_path", type=str, default="./result_Files", help="result files path.")
    arg_s = parser.parse_args()
    return arg_s


def get_eval_result(arg_s):
    max_num = 128
    result_path = arg_s.result_path

    outputs = []

    dataset_coco = COCO(arg_s.anno_path)

    images = dataset_coco.imgs
    for _, v in images.items():
        filename = v['file_name']
        file_id = filename.split('.')[0]

        bbox_result_file = os.path.join(result_path, file_id + "_0.bin")
        label_result_file = os.path.join(result_path, file_id + "_1.bin")
        mask_result_file = os.path.join(result_path, file_id + "_2.bin")

        all_bbox = np.fromfile(bbox_result_file, dtype=np.float16).reshape(20000, 5)
        all_label = np.fromfile(label_result_file, dtype=np.int32).reshape(20000, 1)
        all_mask = np.fromfile(mask_result_file, dtype=np.bool_).reshape(20000, 1)

        all_bbox_squee = np.squeeze(all_bbox)
        all_label_squee = np.squeeze(all_label)
        all_mask_squee = np.squeeze(all_mask)

        all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
        all_labels_tmp_mask = all_label_squee[all_mask_squee]

        if all_bboxes_tmp_mask.shape[0] > max_num:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
            inds = inds[:max_num]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]

        outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
        outputs.append(outputs_tmp)

    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    coco_eval(config, result_files, eval_types, dataset_coco, single_result=False)


if __name__ == '__main__':
    args = prepare_args()
    get_eval_result(args)
