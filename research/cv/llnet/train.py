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
"""train LLNet"""
import os
import time
from math import ceil
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src.dataset import open_mindrecord_dataset
from src.llnet import SDA, SDA_WithLossCell, SDA_TrainOneStepCell, LLNet
from src.lr_generator import get_lr
from src.model_utils.config import config

def prepare_env():
    """
    prepare_env: set the context and config
    """
    set_seed(config.random_seed)

    if config.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)

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
    print("finetrain epoch_size = ", config.finetrain_epoch_size, " train_batch_size = ", config.train_batch_size,
          " lr_init = ", config.lr_init, " weight_decay = ", config.weight_decay)

def generate_pretrain_net(net, lr_init, epoch_size, train_batches_per_epoch, params):
    net.set_train()
    loss_net = SDA_WithLossCell(net)
    # learning rate schedule
    lr = get_lr(lr_init=lr_init, lr_decay_rate=0.1, num_epoch_per_decay=200,
                total_epochs=epoch_size, steps_per_epoch=train_batches_per_epoch)
    lr = Tensor(lr)
    # define optimization
    optimizer = nn.Adam(params=params, learning_rate=lr,
                        weight_decay=config.weight_decay)
    train_net = SDA_TrainOneStepCell(loss_net, optimizer)
    train_net.set_train()
    return train_net

def pretrain(epoch_size, train_batches_per_epoch, train_dataset):
    net1 = SDA(pretrain=True, pretrain_corrupted_level=0.1)
    params1 = net1.trainable_params()
    train_net1 = generate_pretrain_net(net1, config.lr_init[0], epoch_size, train_batches_per_epoch, params1)

    print("SSDA1 pretrain ...")
    for epoch in range(epoch_size):
        step = 0
        steps = train_batches_per_epoch
        train_loss_value = 0
        for data in train_dataset.create_dict_iterator():
            result = train_net1(data["origin"], data["origin"])
            train_loss_value = train_loss_value + result
            step = step + 1
        train_loss_value = train_loss_value /steps
        print(f"Epoch: [{epoch} / {epoch_size}], "
              f"train loss: {train_loss_value} ")

    net1.pretrain = False
    net1.set_train(False)
    net2 = SDA(input_shape=867, output_shape=578, pretrain=True, pretrain_corrupted_level=0.1)
    params2 = net2.trainable_params()
    train_net2 = generate_pretrain_net(net2, config.lr_init[1], epoch_size, train_batches_per_epoch, params2)

    print("SSDA2 pretrain ...")
    for epoch in range(epoch_size):
        step = 0
        steps = train_batches_per_epoch
        train_loss_value = 0
        for data in train_dataset.create_dict_iterator():
            net1_y1 = net1(data["origin"])
            result = train_net2(net1_y1, net1_y1)
            train_loss_value = train_loss_value + result
            step = step + 1
        train_loss_value = train_loss_value /steps
        print(f"Epoch: [{epoch} / {epoch_size}], "
              f"train loss: {train_loss_value} ")

    net1.pretrain = False
    net2.pretrain = False
    net1.set_train(False)
    net2.set_train(False)
    net3 = SDA(input_shape=578, output_shape=289, pretrain=True, pretrain_corrupted_level=0.1)
    params3 = net3.trainable_params()
    train_net3 = generate_pretrain_net(net3, config.lr_init[2], epoch_size, train_batches_per_epoch, params3)

    print("SSDA3 pretrain ...")
    for epoch in range(epoch_size):
        step = 0
        steps = train_batches_per_epoch
        train_loss_value = 0
        for data in train_dataset.create_dict_iterator():
            net1_y1 = net1(data["origin"])
            net2_y1 = net2(net1_y1)
            result = train_net3(net2_y1, net2_y1)
            train_loss_value = train_loss_value + result
            step = step + 1
        train_loss_value = train_loss_value /steps
        print(f"Epoch: [{epoch} / {epoch_size}], "
              f"train loss: {train_loss_value} ")

    net1.pretrain = False
    net2.pretrain = False
    net3.pretrain = False
    net1.set_train()
    net2.set_train()
    net3.set_train()
    return net1, net2, net3

def train():
    start_time = time.time()

    prepare_env()
    device_id = config.device_id
    resume = config.resume
    if config.enable_modelarts:
        # download dataset from obs to server
        import moxing
        print("=========================================================")
        print("config.data_url  =", config.data_url)
        print("config.data_path =", config.data_path)

        moxing.file.copy_parallel(src_url=config.data_url, dst_url=config.data_path)
        print(os.listdir(config.data_path))
        print("=========================================================")

        dataset_path = os.path.join(config.data_path, 'dataset/train')

        # download the checkpoint from obs to server
        if resume != '':
            base_name = os.path.basename(resume)
            dst_url = os.path.join(config.load_path, base_name)
            moxing.file.copy_parallel(src_url=resume, dst_url=dst_url)
            resume = dst_url

        # the path for the output of training
        save_checkpoint_path = os.path.join(config.output_path, str(device_id))
    else:
        dataset_path = config.dataset_path
        save_checkpoint_path = os.path.join(config.save_ckpt_path, 'ckpt_' + str(config.rank) + '/')
    log_filename = os.path.join(save_checkpoint_path, 'log_' + str(device_id) + '.txt')
    # dataloader
    if dataset_path.find('/train') > 0:
        dataset_train_path = dataset_path
    else:
        dataset_train_path = os.path.join(dataset_path, 'train')
    dataset_train_path = os.path.join(dataset_train_path, 'train_1250patches_per_image.mindrecords')
    train_dataset = open_mindrecord_dataset(dataset_train_path, True, config.rank, config.group_size,
                                            columns_list=["noise_darkened", "origin"],
                                            num_parallel_workers=config.work_nums,
                                            batch_size=config.train_batch_size,
                                            drop_remainder=config.drop_remainder, shuffle=True)
    train_batches_per_epoch = train_dataset.get_dataset_size()
    print('train_batches_per_epoch =', train_batches_per_epoch)
    epoch_size = config.pretrain_epoch_size
    net6 = LLNet()
    print("============== Starting Training ==============")
    if resume != '':
        ckpt = load_checkpoint(resume)
        load_param_into_net(net6, ckpt)
        print(resume, ' is loaded')
        net6.da1.set_train()
        net6.da2.set_train()
        net6.da3.set_train()
        net6.da4.set_train()
        net6.da5.set_train()
        net6.da6.set_train()
    else:
        net1, net2, net3 = pretrain(epoch_size, train_batches_per_epoch, train_dataset)
        net6.da1 = net1
        net6.da2 = net2
        net6.da3 = net3
        net6.initial_decoder(w1s=[net3.w1_, net2.w1_, net1.w1_],
                             b1s=[net3.b1_, net2.b1_, net1.b1_],
                             b1_s=[net3.b1, net2.b1, net1.b1])
        net6.da4.set_train()
        net6.da5.set_train()
        net6.da6.set_train()
    params6 = net6.trainable_params()
    epoch_size = config.finetrain_epoch_size
    net6.set_train()
    loss = nn.MSELoss(reduction='mean')
    # learning rate schedule
    lr = get_lr(lr_init=config.lr_init[3], lr_decay_rate=0.1, num_epoch_per_decay=1000,
                total_epochs=epoch_size, steps_per_epoch=train_batches_per_epoch)
    lr = Tensor(lr)
    if resume != '':
        resume_epoch = config.resume_epoch
        lr = lr[train_batches_per_epoch * resume_epoch:]
        epoch_size = epoch_size - resume_epoch
        print('epoch_size is changed to ', epoch_size)
    # define optimization
    optimizer = nn.Adam(params=params6, learning_rate=lr,
                        weight_decay=config.weight_decay)
    if config.device_target == 'Ascend':
        model = Model(net6, loss_fn=loss, optimizer=optimizer, amp_level=config.amp_level)
    else:
        model = Model(net6, loss_fn=loss, optimizer=optimizer)
    loss_cb = LossMonitor(per_print_times=train_batches_per_epoch)
    time_cb = TimeMonitor(data_size=train_batches_per_epoch)
    config_ck = CheckpointConfig(save_checkpoint_steps=train_batches_per_epoch,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"llnet-rank{config.rank}",
                                 directory=save_checkpoint_path, config=config_ck)
    if config.is_distributed and config.save_only_device_0 and device_id != 0:
        callbacks = [loss_cb, time_cb]
    else:
        callbacks = [loss_cb, time_cb, ckpoint_cb]

    model.train(epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=True)

    print("============== Train Success ==================")

    print("data_url   = ", config.data_url)
    print("epoch_size = ", epoch_size, " train_batch_size = ", config.train_batch_size,
          " lr_init = ", config.lr_init, " weight_decay = ", config.weight_decay)
    print("time: ", ceil(time.time() - start_time), " seconds")

    fp = open(log_filename, 'at+')

    print("data_url   = ", config.data_url, file=fp)
    print("epoch_size = ", epoch_size, " train_batch_size = ", config.train_batch_size,
          " lr_init = ", config.lr_init, " weight_decay = ", config.weight_decay, file=fp)

    print("time: ", ceil(time.time() - start_time), " seconds", file=fp)
    fp.close()

if __name__ == '__main__':
    train()

    if config.enable_modelarts:
        import moxing as mox
        print("=========================================================")
        print(os.listdir("/home/work/user-job-dir/"))
        if os.path.exists('/home/work/user-job-dir/outputs'):
            print("config.train_url =", config.train_url)
            mox.file.copy_parallel(src_url='/home/work/user-job-dir/outputs', dst_url=config.train_url)
