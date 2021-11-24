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
"""train imagenet."""
import os
import time

import mindspore
import mindspore.nn as nn

from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.proxylessnas_mobile import proxylessnas_mobile
from src.lr_generator import two_warmup_cosine_annealing_lr

from src.CrossEntropySmooth import CrossEntropySmooth

from src.model_utils.config import config

def prepare_env():
    """
    prepare_env: set the context and config
    """
    print('epoch_size = ', config.epoch_size, ' num_classes = ', config.num_classes)

    set_seed(config.random_seed)

    if config.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
        context.set_context(enable_graph_kernel=config.enable_graph_kernel)

    if config.device_target == 'Ascend':
        context.set_context(enable_reduce_precision=config.enable_reduce_precision)

    # init distributed
    if config.is_distributed:
        init()

        if config.enable_modelarts:
            device_id = get_rank()
            config.group_size = get_group_size()
        else:
            if config.device_target == 'Ascend':
                device_id = int(os.getenv('DEVICE_ID', default='0'))
                config.group_size = int(os.getenv('DEVICE_NUM', default='1'))
            else:
                device_id = get_rank()
                config.group_size = get_group_size()

        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=config.group_size,
                                          gradients_mean=True)
    else:
        device_id = config.device_id
        config.group_size = 1
        context.set_context(device_id=device_id)
    rank_id = device_id

    config.rank = rank_id
    config.device_id = device_id

    print('rank_id = ', rank_id, ' group_size = ', config.group_size)

    print("cutout = ", config.cutout, " cutout_length = ", config.cutout_length)
    print("epoch_size = ", config.epoch_size, " train_batch_size = ", config.train_batch_size,
          " lr_init = ", config.lr_init, " weight_decay = ", config.weight_decay)

def train():
    """
    train: for model trainning
    """
    start_time = time.time()

    device_id = config.device_id
    resume = config.resume

    if config.enable_modelarts:
        # download dataset from obs to cache
        import moxing
        dataset_path = '/cache/dataset'
        if config.data_url.find('/train/') > 0:
            dataset_path += '/train/'
        moxing.file.copy_parallel(src_url=config.data_url, dst_url=dataset_path)

        # download the checkpoint from obs to cache
        if resume != '':
            base_name = os.path.basename(resume)
            dst_url = '/cache/checkpoint/' + base_name
            moxing.file.copy_parallel(src_url=resume, dst_url=dst_url)
            resume = dst_url

        # the path for the output of training
        save_checkpoint_path = '/cache/train_output/' + str(device_id) + '/'
    else:
        dataset_path = config.dataset_path
        save_checkpoint_path = os.path.join(config.save_ckpt_path, 'ckpt_' + str(config.rank) + '/')

    log_filename = os.path.join(save_checkpoint_path, 'log_' + str(device_id) + '.txt')

    # dataloader
    if dataset_path.find('/train') > 0:
        dataset_train_path = dataset_path
    else:
        dataset_train_path = os.path.join(dataset_path, 'train')

    train_dataset = create_dataset(dataset_train_path, True, config.rank, config.group_size,
                                   num_parallel_workers=config.work_nums,
                                   batch_size=config.train_batch_size,
                                   drop_remainder=config.drop_remainder, shuffle=True,
                                   cutout=config.cutout, cutout_length=config.cutout_length)
    train_batches_per_epoch = train_dataset.get_dataset_size()

    # network
    net = proxylessnas_mobile(num_classes=config.num_classes)
    net.set_train()

    # learning rate schedule
    epoch_size = config.epoch_size

    lr = two_warmup_cosine_annealing_lr(lr=config.lr_init, max_epoch=epoch_size,
                                        steps_per_epoch=train_batches_per_epoch,
                                        warmup_epochs=5, T_max=epoch_size, eta_min=0.0)
    if resume != '':
        ckpt = load_checkpoint(resume)
        load_param_into_net(net, ckpt)
        print(resume, ' is loaded')

        resume_epoch = config.resume_epoch
        lr = lr[train_batches_per_epoch * resume_epoch:]
        epoch_size = epoch_size - resume_epoch
        print('epoch_size is changed to ', epoch_size)
    lr = Tensor(lr)

    # define optimization
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    optimizer = nn.SGD(params=group_params, learning_rate=lr, momentum=config.momentum,
                       weight_decay=config.weight_decay)

    # define loss
    if config.enable_label_smooth:
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net.set_train()

    print('scale_factor = ', config.scale_factor)
    print('scale_window = ', config.scale_window)
    loss_scale_manager = mindspore.DynamicLossScaleManager(scale_factor=config.scale_factor,
                                                           scale_window=config.scale_window)
    model = Model(net, loss_fn=loss, optimizer=optimizer, amp_level=config.amp_level, metrics={'acc'},
                  loss_scale_manager=loss_scale_manager)

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=train_batches_per_epoch)
    time_cb = TimeMonitor(data_size=train_batches_per_epoch)

    config_ck = CheckpointConfig(save_checkpoint_steps=train_batches_per_epoch,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"proxylessnas-mobile-rank{config.rank}",
                                 directory=save_checkpoint_path, config=config_ck)

    callbacks = [loss_cb, time_cb, ckpoint_cb]

    model.train(epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    print("============== Train Success ==================")

    print("data_url   = ", config.data_url)
    print("cutout = ", config.cutout, " cutout_length = ", config.cutout_length)
    print("epoch_size = ", epoch_size, " train_batch_size = ", config.train_batch_size,
          " lr_init = ", config.lr_init, " weight_decay = ", config.weight_decay)

    print("time: ", (time.time() - start_time) / 3600, " hours")

    fp = open(log_filename, 'at+')

    print("data_url   = ", config.data_url, file=fp)
    print("cutout = ", config.cutout, " cutout_length = ", config.cutout_length, file=fp)
    print("epoch_size = ", epoch_size, " train_batch_size = ", config.train_batch_size,
          " lr_init = ", config.lr_init, " weight_decay = ", config.weight_decay, file=fp)

    print("time: ", (time.time() - start_time) / 3600, file=fp)
    fp.close()

if __name__ == '__main__':
    prepare_env()
    train()
    if config.enable_modelarts:
        if os.path.exists('/cache/train_output'):
            moxing.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)
