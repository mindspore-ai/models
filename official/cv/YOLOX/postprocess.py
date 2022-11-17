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
# =======================================================================================
"""
post-precess for inference
"""
import os

import numpy as np
from tqdm import tqdm

from model_utils.config import config
from src.util import DetectionEngine


def calculate_coco_ap():
    """
    calculate coco ap
    """
    result_path = config.result_path
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    detection = DetectionEngine(config)
    files = os.listdir(result_path)
    total_imgs = len(files)
    print("Start to inference, totally {} to eval".format(total_imgs))
    grid_size = [(config.input_size[0] / x) * (config.input_size[1] / x) for x in
                 config.fpn_strides]
    anchors = int(sum(grid_size))
    attr_num = config.num_classes + 5
    for file in tqdm(files, desc="Image generate progress",
                     total=total_imgs, unit="img", colour="GREEN"):
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            output = np.fromfile(full_file_path, dtype=np.float32).reshape(
                (-1, anchors, attr_num))
            img_id, img_info_0, img_info_1, _ = file.split(".")[0].split("_")
            detection.detection(output, np.array([[int(img_info_0), int(img_info_1)]]), np.array([[img_id]]))
    print('Start to calculate mAP...')
    result_file_path = detection.evaluate_prediction()
    print('result file path: %s', result_file_path)
    eval_result, _ = detection.get_eval_result()
    eval_print_str = '\n=============coco eval result=========\n' + eval_result
    print(eval_print_str)


if __name__ == '__main__':
    calculate_coco_ap()
