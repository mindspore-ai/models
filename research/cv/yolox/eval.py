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
from tqdm import tqdm
from model_utils.config import config
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from src.logger import get_logger
from src.util import DetectionEngine
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
    rank_id = int(os.getenv('RANK_ID', '0'))
    config.logger = get_logger(config.outputs_dir, rank_id)

    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)
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
        param_dict = load_checkpoint(config.val_ckpt)
        ema_param_dict = {}
        for param in param_dict:
            if param.startswith("ema."):
                new_name = param.split("ema.")[1]
                data = param_dict[param]
                data.name = new_name
                ema_param_dict[new_name] = data

        load_param_into_net(network, ema_param_dict)
        config.logger.info('load model %s success', config.val_ckpt)
    else:
        config.logger.info('%s doesn''t exist or is not a pre-trained file', config.val_ckpt)
        raise FileNotFoundError('{} not exist or not a pre-trained file'.format(config.val_ckpt))
    data_root = config.data_root
    anno_file = config.annFile
    ds = create_yolox_dataset(data_root, anno_file, is_training=False, batch_size=config.per_batch_size, device_num=1,
                              rank=rank_id)
    data_size = ds.get_dataset_size()
    config.logger.info(
        'Finish loading the dataset, totally %s images to eval, iters %s' % (data_size * config.per_batch_size, \
                                                                                 data_size))
    network.set_train(False)
    # init detection engine
    detection = DetectionEngine(config)
    config.logger.info('Start inference...')
    for _, data in enumerate(
            tqdm(ds.create_dict_iterator(num_epochs=1), total=data_size,
                 colour="GREEN")):
        image = data['image']
        img_info = data['image_shape']
        img_id = data['img_id']
        prediction = network(image)
        prediction = prediction.asnumpy()
        img_shape = img_info.asnumpy()
        img_id = img_id.asnumpy()
        detection.detection(prediction, img_shape, img_id)

    config.logger.info('Calculating mAP...')
    result_file_path = detection.evaluate_prediction()
    config.logger.info('result file path: %s', result_file_path)
    eval_result, _ = detection.get_eval_result()
    eval_print_str = '\n=============coco eval result=========\n' + eval_result
    config.logger.info(eval_print_str)


if __name__ == '__main__':
    run_test()
