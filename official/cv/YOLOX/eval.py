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
from model_utils.config import config
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank

from src.logger import get_logger
from src.util import load_weights, EvalWrapper, DetectionEngine
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
    config.log_dir = os.path.join(
        config.log_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')
    )
    config.eval_parallel = config.is_distributed and config.eval_parallel
    rank_id = int(os.getenv('RANK_ID', '0'))
    config.logger = get_logger(config.log_dir, rank_id)
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
    network = DetectionBlock(config, backbone=backbone)  # default yolox-darknet53
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
    ds = create_yolox_dataset(
        data_root, anno_file, is_training=False,
        batch_size=config.per_batch_size,
        device_num=device_num, rank=rank_id
    )
    data_size = ds.get_dataset_size()
    config.logger.info(
        f"Finish loading the dataset, "
        f"totally {data_size * config.per_batch_size} images to eval, iters {data_size}"
    )
    network.set_train(False)
    # init detection engine
    detection = DetectionEngine(config)
    save_prefix = None
    if config.eval_parallel:
        save_prefix = config.eval_parallel_dir
        if os.path.exists(save_prefix):
            shutil.rmtree(save_prefix, ignore_errors=True)
    eval_wrapper = EvalWrapper(
        config=config,
        dataset=ds,
        network=network,
        detection_engine=detection,
        save_prefix=save_prefix
    )
    config.logger.info('Start inference...')
    eval_print_str, _ = eval_wrapper.inference()
    config.logger.info(eval_print_str)
    config.logger.info('Finish inference...')


if __name__ == '__main__':
    run_test()
