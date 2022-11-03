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
for evaluate
"""
import os
import datetime
import shutil
import numpy as np
from model_utils.config import config
import mindspore
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank

from src.logger import get_logger
from src.util import DetectionEngine, load_weights, AllReduce
from src.yolox import DetectionBlock
from src.yolox_dataset import create_yolox_dataset
from src.initializer import default_recurisive_init


def run_test():
    """The function of eval"""
    config.data_root = os.path.join(config.data_dir, 'val2017')
    config.annFile = os.path.join(config.data_dir, 'annotations/instances_val2017.json')

    devid = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=devid)

    # logger
    config.outputs_dir = os.path.join(
        config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')
    )
    config.eval_parallel = config.is_distributed and config.eval_parallel
    rank_id = int(os.getenv('RANK_ID', '0'))
    config.logger = get_logger(config.outputs_dir, rank_id)
    parallel_mode = ParallelMode.STAND_ALONE
    device_num = 1
    if config.eval_parallel:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
        device_num = get_group_size()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=device_num)
    # ------------------network create----------------------------------------------------------------------------
    config.logger.info('Begin Creating Network....')
    if config.backbone == "yolox_darknet53":
        backbone = "yolofpn"
    else:
        backbone = "yolopafpn"
    network = DetectionBlock(config, backbone=backbone)  # default yolo-darknet53
    default_recurisive_init(network)
    config.logger.info(config.val_ckpt)
    if os.path.isfile(config.val_ckpt):
        network = load_weights(network, config.val_ckpt)
        config.logger.info('load model %s success', config.val_ckpt)
    else:
        config.logger.info('%s doesn''t exist or is not a pre-trained file', config.val_ckpt)
        raise FileNotFoundError('{} not exist or not a pre-trained file'.format(config.val_ckpt))
    data_root = config.data_root
    anno_file = config.annFile
    ds = create_yolox_dataset(data_root, anno_file, is_training=False, batch_size=config.per_batch_size,
                              device_num=device_num, rank=rank_id)
    data_size = ds.get_dataset_size()
    config.logger.info(
        'Finish loading the dataset, totally %s images to eval, iters %s' % (data_size * config.per_batch_size, \
                                                                                 data_size))
    network.set_train(False)
    # init detection engine
    if config.eval_parallel:
        save_prefix = config.eval_parallel_dir
        if os.path.exists(save_prefix):
            shutil.rmtree(save_prefix, ignore_errors=True)
        detection = DetectionEngine(config, save_prefix)
        reduce = AllReduce()
    else:
        detection = DetectionEngine(config)
    config.logger.info('Start inference...')
    img_ids = []
    for idx, data in enumerate(ds.create_dict_iterator(num_epochs=1), start=1):
        image = data['image']
        img_shape = data['image_shape']
        img_id = data['img_id']
        prediction = network(image)
        prediction = prediction.asnumpy()
        img_shape = img_shape.asnumpy()
        img_id = img_id.asnumpy()
        if config.eval_parallel:
            mask = np.isin(img_id.squeeze(), img_ids, invert=True)
            prediction, img_shape, img_id = prediction[mask], img_shape[mask], img_id[mask]
            img_ids.extend(img_id.tolist())
        if prediction.shape[0] > 0:
            detection.detection(prediction, img_shape, img_id)
        config.logger.info(f"Detection {idx} / {data_size}")

    config.logger.info('Calculating mAP...')
    if config.eval_parallel:
        result_file_path = detection.evaluate_prediction(cur_epoch=0, cur_step=0, rank_id=rank_id)
    else:
        result_file_path = detection.evaluate_prediction()
    config.logger.info('result file path: %s', result_file_path)
    if config.eval_parallel:
        sync = mindspore.Tensor(np.array([1]).astype(np.int32))
        sync = reduce(sync)
        sync = sync.asnumpy()[0]
        if sync != device_num:
            raise ValueError(f"Sync value {sync} is not equal to number of device {device_num}. "
                             f"There might be wrong with devices.")
        eval_result, _ = detection.get_eval_result_parallel()
    else:
        eval_result, _ = detection.get_eval_result()
    eval_print_str = '\n=============coco eval result=========\n'
    if eval_result is not None:
        eval_print_str = eval_print_str + eval_result
    config.logger.info(eval_print_str)


if __name__ == '__main__':
    run_test()
