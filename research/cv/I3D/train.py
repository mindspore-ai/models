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
"""
Train I3D and save network model files(.ckpt)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random

import numpy as np
import mindspore
from mindspore import context, Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, LearningRateScheduler
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

import src.data_factory as data_factory
import src.model_factory as model_factory
from src.transforms.spatial_transforms import Compose, RandomHorizontalFlip, RandomCrop, CenterCrop
from src.transforms.target_transforms import ClassLabel
from src.transforms.temporal_transforms import TemporalRandomCrop
from src.utils import print_config, write_config, prepare_output_dirs, get_optimizer
from config import parse_opts


def run():
    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
    tic = time.time()
    config = parse_opts()
    if config.dataset == 'ucf101':
        config.finetune_num_classes = 101
    mindspore.set_seed(2022)
    random.seed(2022)
    np.random.seed(2022)
    mindspore.dataset.config.set_seed(2022)
    if config.distributed:
        config.save_dir = './output_distribute/'

    if config.openI:
        import moxing as mox
        obs_data_url = config.data_url
        config.data_url = 'cache/user-job-dir/inputs/data/'
        obs_train_url = config.train_url
        config.train_url = 'cache/user-job-dir/outputs/model/'

        mox.file.copy_parallel(obs_data_url, config.data_url)
        print("Successfully Download {} to {}".format(obs_data_url, config.data_url))

        config.save_dir = config.train_url
        config.video_path = os.path.join(config.data_url, config.dataset, 'jpg')
        if config.mode == 'rgb':
            config.checkpoint_path = os.path.join(os.path.abspath(__file__).replace('train.py', ''),
                                                  'src/pretrained/rgb_imagenet.ckpt')
        if config.mode == 'flow':
            config.checkpoint_path = os.path.join(os.path.abspath(__file__).replace('train.py', ''),
                                                  'src/pretrained/flow_imagenet.ckpt')
        if config.dataset == 'ucf101':
            config.annotation_path = os.path.join(config.data_url, config.dataset, 'annotation/ucf101_01.json')
        if config.dataset == 'hmdb51':
            config.annotation_path = os.path.join(config.data_url, config.dataset, 'annotation/hmdb51_1.json')

    config = prepare_output_dirs(config)
    print_config(config)
    write_config(config, os.path.join(config.save_dir, 'config.json'))

    assert config.context in ['py', 'gr']
    if config.distributed:
        if config.context == 'py':
            context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                                device_id=int(os.environ["DEVICE_ID"]))
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                                device_id=int(os.environ["DEVICE_ID"]))
        config.device_id = int(os.environ["DEVICE_ID"])
        init()
        context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        if config.context == 'py':
            context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                                device_id=config.device_id)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                                device_id=config.device_id)

    train_transforms = {'spatial': Compose([RandomCrop(config.spatial_size), RandomHorizontalFlip()]),
                        'temporal': TemporalRandomCrop(config.train_sample_duration),
                        'target': ClassLabel()}
    validation_transforms = {'spatial': Compose([CenterCrop(config.spatial_size)]),
                             'temporal': TemporalRandomCrop(config.test_sample_duration),
                             'target': ClassLabel()}

    model, parameters = model_factory.get_model(config)
    model.set_train()
    optimizer = get_optimizer(config, parameters, config.lr)
    criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    dataset = data_factory.get_dataset(config, train_transforms, validation_transforms)
    step_size = dataset['train'].get_dataset_size()
    print('step size per epoch:', step_size)
    lr_de_steps = step_size * config.lr_de_epochs

    def learning_rate_function(lr, cur_step_num):
        if not config.has_back:
            if config.mode == 'flow' and cur_step_num % lr_de_steps == 0 and lr <= 1e-5 * 5:
                lr = 0.001
                config.has_back = True
            if config.mode == 'flow' and cur_step_num % lr_de_steps == 0 and lr > 1e-5 * 5:
                lr = lr * config.lr_de_rate
        if config.mode == 'flow' and cur_step_num % lr_de_steps == 0 and config.has_back and lr > 1e-5 * 5:
            lr = lr * config.lr_de_rate

        if config.mode == 'rgb' and cur_step_num % lr_de_steps == 0 and lr > 1e-5:
            lr = lr * config.lr_de_rate

        return lr

    if config.mode == 'rgb':
        config.checkpoints_num_keep = config.checkpoints_num_keep / 2
    loss_scale_manager = DynamicLossScaleManager()
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    lr_cb = LearningRateScheduler(learning_rate_function)
    config_ck = CheckpointConfig(save_checkpoint_steps=config.checkpoint_frequency * step_size,
                                 keep_checkpoint_max=int(config.checkpoints_num_keep))
    ckpt_cb = ModelCheckpoint(prefix="i3d", directory=config.checkpoint_dir, config=config_ck)
    cb = [time_cb, loss_cb, lr_cb, ckpt_cb]

    model = Model(network=model, loss_fn=criterion, optimizer=optimizer, amp_level=config.amp_level,
                  loss_scale_manager=loss_scale_manager)
    model.train(epoch=config.num_epochs, train_dataset=dataset['train'], callbacks=cb,
                dataset_sink_mode=config.sink_mode)

    toc = time.time()
    total_time = toc - tic
    print('total_time:', total_time)
    print('Finished training.')

    if config.openI:
        mox.file.copy_parallel(config.train_url, obs_train_url)
        print("Successfully Upload {} to {}".format(config.train_url, obs_train_url))


if __name__ == '__main__':
    run()
