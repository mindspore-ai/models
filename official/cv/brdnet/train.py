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
import os
import datetime
from pprint import pprint

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.common.tensor import Tensor
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import SummaryCollector

from src.logger import get_logger
from src.dataset import create_BRDNetDataset
from src.models import BRDNet, BRDWithLossCell, TrainingWrapper
from src.config import config as cfg


def get_lr(steps_per_epoch, max_epoch, init_lr):
    lr_each_step = []
    for step in range(steps_per_epoch*max_epoch):
        if step < (steps_per_epoch*30):
            lr_each_step.append(init_lr)
        else:  # decrease lr after first 30 epochs
            lr_each_step.append(init_lr/10)
    return lr_each_step


def train():

    save_dir = os.path.join(cfg.output_path, 'sigma_' + str(cfg.sigma) \
            + '_' + datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    summary_dir = os.path.join(save_dir, 'summary')

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, save_graphs=False)

    if cfg.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
    elif cfg.device_target != "GPU":
        raise NotImplementedError("Training only supported for CPU and GPU.")

    if cfg.is_distributed:
        if cfg.device_target == "Ascend":
            init()
            cfg.rank = get_rank()
            cfg.group_size = get_group_size()
            device_num = cfg.group_size
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL)
        elif cfg.device_target == "GPU":
            init('nccl')
            cfg.rank = get_rank()
            cfg.group_size = get_group_size()
            device_num = cfg.group_size
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)

    if cfg.rank == 0:
        pprint(cfg)
        print("Please check the above configuration\n")
        if not cfg.use_modelarts and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    cfg.rank_save_ckpt_flag = 0
    if cfg.is_save_on_master:
        if cfg.rank == 0:
            cfg.rank_save_ckpt_flag = 1
    else:
        cfg.rank_save_ckpt_flag = 1

    cfg.logger = get_logger(save_dir, "BRDNet", cfg.rank)
    cfg.logger.save_args(cfg)

    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying train data from obs to cache....")
        mox.file.copy_parallel(cfg.train_data, 'cache/dataset')
        cfg.logger.info("copying traindata finished....")
        cfg.train_data = 'cache/dataset/'

    dataset, cfg.steps_per_epoch = create_BRDNetDataset(cfg.train_data, cfg.sigma, \
                        cfg.channel, cfg.batch_size, cfg.group_size, cfg.rank, shuffle=True)
    model = BRDNet(cfg.channel)

    # resume checkpoint if needed
    if cfg.resume_path:
        if cfg.use_modelarts:
            import moxing as mox
            cfg.logger.info("copying resume checkpoint from obs to cache....")
            mox.file.copy_parallel(cfg.resume_path, 'cache/resume_path')
            cfg.logger.info("copying resume checkpoint finished....")
            cfg.resume_path = 'cache/resume_path/'

        cfg.resume_path = os.path.join(cfg.resume_path, cfg.resume_name)
        cfg.logger.info('loading resume checkpoint %s into network', str(cfg.resume_path))
        load_param_into_net(model, load_checkpoint(cfg.resume_path))
        cfg.logger.info('loaded resume checkpoint %s into network', str(cfg.resume_path))


    model = BRDWithLossCell(model)
    model.set_train()

    lr_list = get_lr(cfg.steps_per_epoch, cfg.epoch, cfg.lr)
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=Tensor(lr_list, mindspore.float32))
    model = TrainingWrapper(model, optimizer)

    model = Model(model)

    # define callbacks
    if cfg.rank == 0:
        time_cb = TimeMonitor(data_size=cfg.steps_per_epoch)
        loss_cb = LossMonitor(per_print_times=10)
        summary_cb = SummaryCollector(summary_dir)
        callbacks = [time_cb, loss_cb, summary_cb]
    else:
        callbacks = []
    if cfg.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.steps_per_epoch*cfg.save_every,
                                       keep_checkpoint_max=cfg.ckpt_save_max)
        save_ckpt_path = os.path.join(save_dir, 'ckpt_' + str(cfg.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='channel_'+str(cfg.channel)+'_sigma_'+str(cfg.sigma)+'_rank_'+str(cfg.rank))
        callbacks.append(ckpt_cb)

    print(f"[rank: {cfg.rank}] Model configured, start training\n")
    model.train(cfg.epoch, dataset, callbacks=callbacks, dataset_sink_mode=True)

    cfg.logger.info("training finished....")
    if cfg.use_modelarts:
        cfg.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(save_dir, cfg.outer_path)
        cfg.logger.info("copying finished....")

if __name__ == '__main__':
    set_seed(1)

    print("Entering train...")
    train()
    cfg.logger.info('All task finished!')
