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
"""YoloV5 310 infer."""
import os
import time
import numpy as np
from pycocotools.coco import COCO
from src.logger import get_logger
from src.util import DetectionEngine
from model_utils.config import config


if __name__ == "__main__":
    start_time = time.time()
    config.output_dir = config.log_path
    config.logger = get_logger(config.output_dir, 0)

    # init detection engine
    detection = DetectionEngine(config, config.test_ignore_threshold)

    coco = COCO(config.ann_file)
    result_path = config.result_files

    files = os.listdir(config.dataset_path)

    for file in files:
        img_ids_name = file.split('.')[0]
        img_id_ = int(np.squeeze(img_ids_name))
        imgIds = coco.getImgIds(imgIds=[img_id_])
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        image_shape = ((img['width'], img['height']),)
        img_id_ = (np.squeeze(img_ids_name),)

        result_path_0 = os.path.join(result_path, img_ids_name + "_0.bin")
        result_path_1 = os.path.join(result_path, img_ids_name + "_1.bin")
        result_path_2 = os.path.join(result_path, img_ids_name + "_2.bin")

        output_small = np.fromfile(result_path_0, dtype=np.float32).reshape(1, 20, 20, 3, 85)
        output_me = np.fromfile(result_path_1, dtype=np.float32).reshape(1, 40, 40, 3, 85)
        output_big = np.fromfile(result_path_2, dtype=np.float32).reshape(1, 80, 80, 3, 85)

        detection.detect([output_small, output_me, output_big], config.batch_size, image_shape, img_id_)

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    config.logger.info('result file path: %s', result_file_path)
    eval_result = detection.get_eval_result()

    cost_time = time.time() - start_time
    config.logger.info('=============coco 310 infer reulst=========')
    config.logger.info(eval_result)
    config.logger.info('testing cost time {:.2f}h'.format(cost_time / 3600.))
