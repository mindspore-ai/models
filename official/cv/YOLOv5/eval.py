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
"""YoloV5 eval."""
import os
import time

import numpy as np

import mindspore as ms

from src.yolo import YOLOV5
from src.logger import get_logger
from src.util import DetectionEngine
from src.yolo_dataset import create_yolo_dataset

from model_utils.config import config

# only useful for huawei cloud modelarts
from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process


def eval_preprocess():
    config.data_root = os.path.join(config.data_dir, 'val2017')
    config.ann_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    device_id = int(os.getenv('DEVICE_ID', '0'))
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

    # logger module is managed by config, it is used in other function. e.x. config.logger.info("xxx")
    config.logger = get_logger(config.output_dir, device_id)


def load_parameters(network, filename):
    config.logger.info("yolov5 pretrained network model: %s", filename)
    param_dict = ms.load_checkpoint(filename)
    param_dict_new = {}
    for key, values in param_dict.items():
        if key.startswith('moments.'):
            continue
        elif key.startswith('yolo_network.'):
            param_dict_new[key[13:]] = values
        else:
            param_dict_new[key] = values
    ms.load_param_into_net(network, param_dict_new)
    config.logger.info('load_model %s success', filename)


@moxing_wrapper(pre_process=modelarts_pre_process, pre_args=[config])
def run_eval():
    eval_preprocess()
    start_time = time.time()
    config.logger.info('Creating Network....')
    dict_version = {'yolov5s': 0, 'yolov5m': 1, 'yolov5l': 2, 'yolov5x': 3}
    network = YOLOV5(is_training=False, version=dict_version[config.yolov5_version])

    if os.path.isfile(config.pretrained):
        load_parameters(network, config.pretrained)
    else:
        raise FileNotFoundError(f"{config.pretrained} is not a filename.")

    ds = create_yolo_dataset(config.data_root, config.ann_file, is_training=False, batch_size=config.per_batch_size,
                             device_num=1, rank=0, shuffle=False, config=config)

    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('total %d images to eval', ds.get_dataset_size() * config.per_batch_size)

    network.set_train(False)

    # init detection engine
    detection = DetectionEngine(config, config.test_ignore_threshold)

    input_shape = ms.Tensor(tuple(config.test_img_shape), ms.float32)
    config.logger.info('Start inference....')
    for index, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        image = data["image"]
        # adapt network shape of input data
        image = np.concatenate((image[..., ::2, ::2], image[..., 1::2, ::2],
                                image[..., ::2, 1::2], image[..., 1::2, 1::2]), axis=1)
        image = ms.Tensor(image)
        image_shape_ = data["image_shape"]
        image_id_ = data["img_id"]
        output_big, output_me, output_small = network(image, input_shape)
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape_, image_id_)

        if index % 50 == 0:
            config.logger.info('Processing... {:.2f}% '.format(index / ds.get_dataset_size() * 100))

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
