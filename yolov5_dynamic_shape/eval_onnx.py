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
"""Run evaluation for a model exported to ONNX"""

import datetime
import os
import time

import numpy as np
import onnxruntime as ort
from mindspore.context import ParallelMode
from mindspore import context

from eval import DetectionEngine
from model_utils.config import config
from src.logger import get_logger
from src.yolo_dataset import create_yolo_dataset


def create_session(checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )

    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


def run_eval():
    """Run yolov5 eval"""
    start_time = time.time()

    session, input_names = create_session(config.file_name, config.device_target)

    config.rank = 0
    config.data_root = os.path.join(config.data_dir, 'val2017')
    config.ann_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')

    data_root = config.data_root
    ann_file = config.ann_file

    config.outputs_dir = os.path.join(config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.getenv('DEVICE_ID', '0'))
    config.logger = get_logger(config.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    ds = create_yolo_dataset(data_root, ann_file, batch_size=config.per_batch_size, device_num=1,
                             rank=rank_id, config=config, is_training=False, shuffle=False)

    config.logger.info('testing shape : %s', config.test_img_shape)

    # init detection engine
    detection = DetectionEngine(config, config.test_ignore_threshold)

    input_shape = np.array(config.test_img_shape, dtype=np.int64)
    config.logger.info('Start inference....')
    for image_index, data in enumerate(ds.create_dict_iterator(num_epochs=1)):
        image = data["image"].asnumpy()
        image = np.concatenate((image[..., ::2, ::2], image[..., 1::2, ::2],
                                image[..., ::2, 1::2], image[..., 1::2, 1::2]), axis=1)
        image_shape_ = data["image_shape"].asnumpy()
        image_id_ = data["img_id"].asnumpy()

        inputs = dict(zip(input_names, (image, input_shape)))
        prediction = session.run(None, inputs)

        output_big, output_me, output_small = prediction
        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape_, image_id_)
        if image_index % 1000 == 0:
            config.logger.info('Processing... {:.2f}% '.format(image_index / ds.get_dataset_size() * 100))

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    config.logger.info('result file path: %s', result_file_path)
    eval_result = detection.get_eval_result()

    cost_time = time.time() - start_time
    eval_log_string = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_log_string)
    config.logger.info('testing cost time %.2f h', cost_time / 3600.)


if __name__ == "__main__":
    run_eval()
