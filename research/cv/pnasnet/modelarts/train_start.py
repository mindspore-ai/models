# Copyright (c) 2022. Huawei Technologies Co., Ltd
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
import argparse
import glob
import ast
import moxing
import numpy as np

import mindspore as ms
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from mindspore.nn.optim.rmsprop import RMSProp

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model

from mindspore.common import set_seed
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.pnasnet_mobile import PNASNet5_Mobile
from src.dataset import create_dataset
from src.pnasnet_mobile import PNASNet5_Mobile_WithLoss
from src.lr_generator import get_lr

def model_export(arguments):
    """export air"""
    output_dir = arguments.local_output_dir
    epoch_size_export = str(arguments.epoch_size)
    ckpt_file = glob.glob(output_dir + '/' + '*' + epoch_size_export + '*' + '.ckpt')[0]
    print("ckpt_file: ", ckpt_file)
    net = PNASNet5_Mobile(num_classes=arguments.num_classes)
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    export_file = os.path.join(output_dir, arguments.file_name)
    export(net, input_data, file_name=export_file, file_format=arguments.file_format)
    return 0

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=True, help="")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                        help="Device target, support Ascend, GPU and CPU.")
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend. (Default: None)')
    parser.add_argument("--enable_profiling", type=ast.literal_eval, default=False, help="")
    parser.add_argument("--num_classes", type=int, default=1000, help="")
    parser.add_argument("--rank", type=int, default=0, help="")
    parser.add_argument("--group_size", type=int, default=1, help="")
    parser.add_argument("--save_checkpoint", type=ast.literal_eval, default=True, help="")
    parser.add_argument("--amp_level", type=str, default="O3", help="")
    parser.add_argument("--is_distributed", type=bool, default=False, help="")

    parser.add_argument('--train_url', required=True, default=None, help='obs browser path')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data')
    parser.add_argument('--local_data_dir', type=str, default="/cache")
    parser.add_argument('--local_output_dir', type=str, default="/cache/train_output")
    parser.add_argument('--save_ckpt_path', type=str, default="./")


    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--epoch_size", type=int, default=600, help="")
    parser.add_argument("--resume_epoch", type=int, default=1, help="")
    parser.add_argument("--resume", type=str, default="", help="")
    parser.add_argument("--random_seed", type=int, default=1, help="")
    parser.add_argument("--work_nums", type=int, default=8, help="")
    parser.add_argument("--cutout_length", type=int, default=56, help="")
    parser.add_argument("--cutout", type=bool, default=True, help="")
    parser.add_argument("--train_batch_size", type=int, default=32, help="")
    parser.add_argument("--val_batch_size", type=int, default=125, help="")
    parser.add_argument("--lr_init", type=float, default=0.32, help="")
    parser.add_argument("--lr_decay_rate", type=float, default=0.97, help="")
    parser.add_argument("--num_epoch_per_decay", type=float, default=2.4, help="")
    parser.add_argument("--drop_path_prob", type=float, default=0.5, help="")
    parser.add_argument("--loss_scale", type=int, default=1, help="")
    parser.add_argument("--aux_factor", type=float, default=0.4, help="")
    parser.add_argument("--opt_eps", type=float, default=1.0, help="")
    parser.add_argument("--rmsprop_decay", type=float, default=0.9, help="")
    parser.add_argument("--keep_checkpoint_max", type=int, default=50, help="")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--weight_decay", type=float, default=0.00004, help="")
    parser.add_argument("--use_pynative_mode", type=ast.literal_eval, default=False, help="")

    parser.add_argument("--file_name", type=str, default="pnasnet_mobile", help="the name of air file ")
    parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR",
                        help="file format")

    config = parser.parse_args()

    prepare_env()
    start_time = time.time()

    resume = config.resume

    if config.enable_modelarts:
        # download dataset from obs to cache
        local_data_dir = '/cache/dataset'
        if config.data_url.find('/train/') > 0:
            local_data_dir += '/train/'
        moxing.file.copy_parallel(src_url=config.data_url, dst_url=local_data_dir)

        # download the checkpoint from obs to cache
        if resume != '':
            base_name = os.path.basename(resume)
            dst_url = '/cache/checkpoint/' + base_name
            moxing.file.copy_parallel(src_url=resume, dst_url=dst_url)
            resume = dst_url

        # the path for the output of training
        save_checkpoint_path = '/cache/train_output/' + str(config.device_id) + '/'
    else:
        local_data_dir = config.local_data_dir
        save_checkpoint_path = os.path.join(config.save_ckpt_path, 'ckpt_' + str(config.rank) + '/')

    log_filename = os.path.join(save_checkpoint_path, 'log_' + str(config.device_id) + '.txt')

    # dataloader
    if local_data_dir.find('/train') > 0:
        dataset_train_path = local_data_dir
    else:
        dataset_train_path = os.path.join(local_data_dir, 'train')
    print(dataset_train_path)
    train_dataset = create_dataset(dataset_train_path, True, config.rank, config.group_size,
                                   num_parallel_workers=config.work_nums,
                                   batch_size=config.train_batch_size,
                                   drop_remainder=True, shuffle=True,
                                   cutout=config.cutout, cutout_length=config.cutout_length)
    train_batches_per_epoch = train_dataset.get_dataset_size()

    # network
    net_with_loss = PNASNet5_Mobile_WithLoss(num_classes=config.num_classes)
    if resume != '':
        ckpt = load_checkpoint(resume)
        load_param_into_net(net_with_loss, ckpt)
        print(resume, ' is loaded')
    net_with_loss.set_train()
    epoch_size = config.epoch_size
    # learning rate schedule
    lr = get_lr(lr_init=config.lr_init, lr_decay_rate=config.lr_decay_rate,
                num_epoch_per_decay=config.num_epoch_per_decay, total_epochs=epoch_size,
                steps_per_epoch=train_batches_per_epoch, is_stair=True)
    if resume:
        resume_epoch = config.resume_epoch
        lr = lr[train_batches_per_epoch * resume_epoch:]
        epoch_size = epoch_size - resume_epoch
    lr = Tensor(lr)

    # define optimization
    decayed_params = []
    no_decayed_params = []
    for param in net_with_loss.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net_with_loss.trainable_params()}]
    optimizer = RMSProp(group_params, lr, decay=config.rmsprop_decay, weight_decay=config.weight_decay,
                        momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)

    if config.device_target == 'Ascend':
        model = Model(net_with_loss, optimizer=optimizer, amp_level=config.amp_level)
    else:
        model = Model(net_with_loss, optimizer=optimizer)

    print("============== Starting Training ==============")
    loss_cb = LossMonitor(per_print_times=train_batches_per_epoch)
    time_cb = TimeMonitor(data_size=train_batches_per_epoch)

    config_ck = CheckpointConfig(save_checkpoint_steps=train_batches_per_epoch,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"pnasnet-mobile-rank{config.rank}",
                                 directory=save_checkpoint_path, config=config_ck)

    callbacks = [loss_cb, time_cb, ckpoint_cb]

    try:
        model.train(epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    except KeyboardInterrupt:
        print("!!!!!!!!!!!!!! Train Failed !!!!!!!!!!!!!!!!!!!")
    else:
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
    if not os.path.exists(config.local_output_dir):
        os.mkdir(config.local_output_dir)
    config.local_output_dir = save_checkpoint_path
    model_export(config)
    if config.enable_modelarts:
        if os.path.exists('/cache/train_output'):
            moxing.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)
