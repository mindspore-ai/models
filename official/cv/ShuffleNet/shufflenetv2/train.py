# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""train_imagenet."""
import argparse
import ast
import os
import time

import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.shufflenetv2 import ShuffleNetV2
from src.config import config_gpu, config_ascend
from src.dataset import create_dataset
from src.lr_generator import get_lr_basic
from src.CrossEntropySmooth import CrossEntropySmooth

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='image classification training')
    parser.add_argument('--dataset_path', type=str, default='../imagenet', help='Dataset path')
    parser.add_argument('--resume', type=str, default='',
                        help='resume training with existed checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=1, help='Resume from which epoch')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False,
                        help='distributed training')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                        help='run platform')
    parser.add_argument('--model_size', type=str, default='1.0x',
                        help='ShuffleNetV2 model size parameter')

    parser.add_argument('--device_id', type=int, default=0, help='device id(Default:0)')

    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path for modelarts')
    parser.add_argument('--train_url', type=str, default=None, help='Output path for modelarts')

    parser.add_argument('--use_pynative_mode', type=ast.literal_eval, default=False,
                        help='whether to use pynative mode for device(Default: False)')
    parser.add_argument('--normalize', type=ast.literal_eval, default=True,
                        help='whether to normalize the dataset(Default: True)')

    parser.add_argument('--enable_tobgr', type=ast.literal_eval, default=False,
                        help='whether to use the toBGR()(Default: False)')

    parser.add_argument('--use_nn_default_loss', type=ast.literal_eval, default=True,
                        help='whether to use nn.SoftmaxCrossEntropyWithLogits(Default: True)')

    parser.add_argument('--overwrite_config', type=ast.literal_eval, default=False,
                        help='whether to overwrite the config according to the arguments')
    #when the overwrite_config == True , the following argument will be written to config
    parser.add_argument('--epoch_size', type=int, default=240,
                        help='Epoches for training(default:240 for Ascend)')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--cutout', type=ast.literal_eval, default=True,
                        help='whether to cutout the data for trainning(Default: True)')
    parser.add_argument('--train_batch_size', type=int, default=96, help='batch size for training')
    parser.add_argument('--lr_init', type=float, default=0.499, help='learning rate for training')

    args_opt = parser.parse_args()

    if args_opt.platform == 'GPU':
        config = config_gpu
        drop_remainder = True
    else:
        config = config_ascend
        drop_remainder = False

    if args_opt.overwrite_config:
        config.epoch_size = args_opt.epoch_size
        config.num_classes = args_opt.num_classes
        config.cutout = args_opt.cutout
        config.train_batch_size = args_opt.train_batch_size
        config.lr_init = args_opt.lr_init

    print('epoch_size = ', config.epoch_size, ' num_classes = ', config.num_classes)
    print('train_batch_size = ', config.train_batch_size, ' lr_init = ', config.lr_init)
    print('cutout = ', config.cutout, ' cutout_length =', config.cutout_length)
    print('normalize = ', args_opt.normalize, ' enable_tobgr = ', args_opt.enable_tobgr,
          ' use_nn_default_loss = ', args_opt.use_nn_default_loss)

    set_seed(config.random_seed)

    if args_opt.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.platform)
        print('mode = PYNATIVE_MODE')
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform, save_graphs=False)
        print('mode = GRAPH_MODE')

    # Set mempool block size in PYNATIVE_MODE for improving memory utilization, which will not take effect in GRAPH_MODE
    if context.get_context("mode") == context.PYNATIVE_MODE:
        context.set_context(mempool_block_size="25GB")

    # init distributed
    if args_opt.is_distributed:
        init()

        if args_opt.is_modelarts:
            device_id = get_rank()
            config.group_size = get_group_size()
        else:
            if args_opt.platform == 'Ascend':
                device_id = int(os.getenv('DEVICE_ID', default='0'))
                config.group_size = int(os.getenv('DEVICE_NUM', default='1'))
            else:
                device_id = get_rank()
                config.group_size = get_group_size()

        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=config.group_size,
                                          gradients_mean=True)
    else:
        device_id = args_opt.device_id
        config.group_size = 1
        context.set_context(device_id=device_id)
    rank_id = device_id
    config.rank = rank_id
    print('rank_id = ', rank_id, ' group_size = ', config.group_size)

    resume = args_opt.resume
    if args_opt.is_modelarts:
        # download dataset from obs to cache
        import moxing
        dataset_path = '/cache/dataset'
        if args_opt.data_url.find('/train/') > 0:
            dataset_path += '/train/'
        moxing.file.copy_parallel(src_url=args_opt.data_url, dst_url=dataset_path)

        # download the checkpoint from obs to cache
        if resume != '':
            base_name = os.path.basename(resume)
            dst_url = '/cache/checkpoint/' + base_name
            moxing.file.copy_parallel(src_url=resume, dst_url=dst_url)
            resume = dst_url

        # the path for the output of training
        save_checkpoint_path = '/cache/train_output/' + str(device_id) + '/'
    else:
        dataset_path = args_opt.dataset_path
        save_checkpoint_path = os.path.join(config.ckpt_path, 'ckpt_' + str(config.rank) + '/')

    # dataloader
    if dataset_path.find('/train') > 0:
        dataset_train_path = dataset_path
    else:
        dataset_train_path = os.path.join(dataset_path, 'train')
        if not os.path.exists(dataset_train_path):
            dataset_train_path = dataset_path

    dataset = create_dataset(dataset_train_path, True, config.rank, config.group_size,
                             num_parallel_workers=config.work_nums, batch_size=config.train_batch_size,
                             drop_remainder=drop_remainder, shuffle=True,
                             cutout=config.cutout, cutout_length=config.cutout_length,
                             normalize=args_opt.normalize,
                             enable_tobgr=args_opt.enable_tobgr)
    batches_per_epoch = dataset.get_dataset_size()

    # network
    net = ShuffleNetV2(n_class=config.num_classes, model_size=args_opt.model_size)

    # loss
    if not args_opt.use_nn_default_loss:
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # learning rate schedule
    lr = get_lr_basic(lr_init=config.lr_init, total_epochs=config.epoch_size,
                      steps_per_epoch=batches_per_epoch, is_stair=True)
    epoch_size = config.epoch_size
    if resume != '':
        resume_epoch = args_opt.resume_epoch
        step_num_in_epoch = dataset.get_dataset_size()
        lr = lr[step_num_in_epoch * resume_epoch:]
        epoch_size = config.epoch_size - resume_epoch
        print('Effective epoch_size = ', epoch_size)
    lr = Tensor(lr)

    optimizer = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                         weight_decay=config.weight_decay)

    if resume != '':
        ckpt = load_checkpoint(resume)
        load_param_into_net(net, ckpt)
        print(resume, ' is loaded')
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'})

    print("============== Starting Training ==============")

    loss_cb = LossMonitor(per_print_times=batches_per_epoch)
    time_cb = TimeMonitor(data_size=batches_per_epoch)
    callbacks = [loss_cb, time_cb]

    checkpoint_config = CheckpointConfig(save_checkpoint_steps=batches_per_epoch,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="shufflenet-rank" + str(config.rank), directory=save_checkpoint_path,
                                 config=checkpoint_config)

    if args_opt.is_distributed and config.is_save_on_master == 1:
        if config.rank == 0:
            callbacks.append(ckpoint_cb)
    else:
        callbacks.append(ckpoint_cb)

    try:
        model.train(epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=True)
    except KeyboardInterrupt:
        print("!!!!!!!!!!!!!! Train Failed !!!!!!!!!!!!!!!!!!!")
    else:
        print("============== Train Success ==================")

    if args_opt.is_modelarts:
        if os.path.exists('/cache/train_output'):
            moxing.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)

    print("time: ", (time.time() - start_time) / 3600, " hours")
    print("train success")
