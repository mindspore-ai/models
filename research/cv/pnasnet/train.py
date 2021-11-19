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

from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from mindspore.nn.optim.rmsprop import RMSProp

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.pnasnet_mobile import PNASNet5_Mobile_WithLoss
from src.lr_generator import get_lr

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

    drop_remainder = config.drop_remainder

    train_dataset = create_dataset(dataset_train_path, True, config.rank, config.group_size,
                                   num_parallel_workers=config.work_nums,
                                   batch_size=config.train_batch_size,
                                   drop_remainder=drop_remainder, shuffle=True,
                                   cutout=config.cutout, cutout_length=config.cutout_length)
    train_batches_per_epoch = train_dataset.get_dataset_size()

    # network
    net_with_loss = PNASNet5_Mobile_WithLoss(num_classes=config.num_classes)
    if resume != '':
        ckpt = load_checkpoint(resume)
        load_param_into_net(net_with_loss, ckpt)
        print(resume, ' is loaded')
    net_with_loss.set_train()

    # learning rate schedule
    epoch_size = config.epoch_size
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

if __name__ == '__main__':
    prepare_env()
    train()
    if config.enable_modelarts:
        if os.path.exists('/cache/train_output'):
            moxing.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)
