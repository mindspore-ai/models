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
'''train.py'''
import os
import datetime
import zipfile
import moxing as mox
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import context, export
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.dataset import config
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint

from src.logger import get_logger
from src.dataset import create_VideoDataset
from src.models import get_r2plus1d_model
from src.utils import TempLoss, AccuracyMetric, EvalCallBack
from src.config import config as cfg

def copy_data_from_obs():
    '''copy_data_from_obs'''
    cfg.logger.info("copying dataset from obs to cache....")
    mox.file.copy_parallel(cfg.dataset_root_path, 'cache/dataset_' + str(cfg.rank))
    cfg.logger.info("copying dataset finished....")
    cfg.dataset_root_path = 'cache/dataset_' + str(cfg.rank)
    cfg.logger.info("starting unzip file to cache....")
    zFile = zipfile.ZipFile(os.path.join(cfg.dataset_root_path, cfg.pack_file_name), "r")
    for fileM in zFile.namelist():
        zFile.extract(fileM, cfg.dataset_root_path)
    zFile.close()
    cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, cfg.pack_file_name.split(".")[0])
    cfg.logger.info("unzip finished....")
    if cfg.pretrain_path:
        cfg.logger.info("copying pretrain checkpoint from obs to cache....")
        mox.file.copy_parallel(cfg.pretrain_path, 'cache/pretrain_' + str(cfg.rank))
        cfg.logger.info("copying pretrain checkpoint finished....")
        cfg.pretrain_path = 'cache/pretrain_' + str(cfg.rank)

    if cfg.resume_path:
        cfg.logger.info("copying resume checkpoint from obs to cache....")
        mox.file.copy_parallel(cfg.resume_path, 'cache/resume_path_' + str(cfg.rank))
        cfg.logger.info("copying resume checkpoint finished....")
        cfg.resume_path = 'cache/resume_path_' + str(cfg.rank)

def copy_data_to_obs():
    cfg.logger.info("copying files from cache to obs....")
    mox.file.copy_parallel(cfg.save_dir, cfg.outer_path)
    cfg.logger.info("copying finished....")

def export_models():
    cfg.logger.info("exporting model....")
    net = get_r2plus1d_model(cfg.num_classes, cfg.layer_num)
    param_dict = load_checkpoint(os.path.join(cfg.save_dir, str(cfg.rank) + "_best_map.ckpt"))
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([cfg.export_batch_size, 3, 16, \
                                cfg.image_height, cfg.image_width]), mindspore.float32)
    export(net, input_arr, file_name=os.path.join(cfg.save_dir, str(cfg.rank) + "_best_map"), \
           file_format=cfg.file_format)
    cfg.logger.info("export model finished....")

def get_lr(steps_per_epoch, max_epoch, init_lr):
    lr_each_step = []
    while max_epoch > 0:
        tem = min(10, max_epoch)
        for _ in range(steps_per_epoch*tem):
            lr_each_step.append(init_lr)
        max_epoch -= tem
        init_lr /= 10
    return lr_each_step

def train():
    '''train'''
    train_dataset, cfg.steps_per_epoch = create_VideoDataset(cfg.dataset_root_path, cfg.dataset_name, \
                      mode='train', clip_len=16, batch_size=cfg.batch_size, \
                      device_num=cfg.group_size, rank=cfg.rank, shuffle=True)
    cfg.logger.info("cfg.steps_per_epoch: %s", str(cfg.steps_per_epoch))
    f_model = get_r2plus1d_model(cfg.num_classes, cfg.layer_num)

    if cfg.pretrain_path and not cfg.resume_path:
        # we execute either pretrain or resume
        cfg.pretrain_path = os.path.join(cfg.pretrain_path, cfg.ckpt_name)
        cfg.logger.info('loading pretrain checkpoint %s into network', str(cfg.pretrain_path))
        param_dict = load_checkpoint(cfg.pretrain_path)
        del param_dict['fc.weight']
        del param_dict['fc.bias']
        load_param_into_net(f_model, param_dict)
        cfg.logger.info('loaded pretrain checkpoint %s into network', str(cfg.pretrain_path))

    # resume checkpoint if needed
    if cfg.resume_path:
        cfg.resume_path = os.path.join(cfg.resume_path, cfg.resume_name)
        cfg.logger.info('loading resume checkpoint %s into network', str(cfg.resume_path))
        load_param_into_net(f_model, load_checkpoint(cfg.resume_path))
        cfg.logger.info('loaded resume checkpoint %s into network', str(cfg.resume_path))

    f_model.set_train()

    # lr scheduling
    lr_list = get_lr(cfg.steps_per_epoch, cfg.epochs, cfg.lr)
    lr_list = lr_list[cfg.steps_per_epoch*cfg.resume_epoch:]
    # optimizer
    optimizer = nn.SGD(params=f_model.trainable_params(), momentum=cfg.momentum,
                       learning_rate=Tensor(lr_list, mindspore.float32), weight_decay=cfg.weight_decay)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')
    model = Model(f_model, loss_fn, optimizer, amp_level="auto")
    # define callbacks
    callbacks = []
    if cfg.rank == 0:
        time_cb = TimeMonitor(data_size=cfg.steps_per_epoch)
        loss_cb = LossMonitor(10)
        callbacks = [time_cb, loss_cb]
    if cfg.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.steps_per_epoch*cfg.save_every,
                                       keep_checkpoint_max=cfg.ckpt_save_max)
        save_ckpt_path = os.path.join(cfg.save_dir, 'ckpt_' + str(cfg.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='rank_'+str(cfg.rank))
        callbacks.append(ckpt_cb)

    if cfg.eval_while_train:
        loss_f = TempLoss()
        val_dataloader, val_data_size = create_VideoDataset(cfg.dataset_root_path, cfg.dataset_name, \
                        mode=cfg.val_mode, clip_len=16, batch_size=cfg.batch_size, \
                        device_num=1, rank=0, shuffle=False)
        network_eval = Model(f_model, loss_fn=loss_f, metrics={"AccuracyMetric": \
                             AccuracyMetric(val_data_size*cfg.batch_size)})
        eval_cb = EvalCallBack(network_eval, val_dataloader, interval=cfg.eval_steps, \
                               eval_start_epoch=max(0, cfg.eval_start_epoch-cfg.resume_epoch), \
                               ckpt_directory=cfg.save_dir, save_best_ckpt=True, \
                               besk_ckpt_name=str(cfg.rank)+"_best_map.ckpt", \
                               f_model=f_model)
        callbacks.append(eval_cb)

    model.train(cfg.epochs-cfg.resume_epoch, train_dataset, callbacks=callbacks, dataset_sink_mode=False)
    cfg.logger.info("training finished....")

if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.join(cfg.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, save_graphs=False)
    if cfg.is_distributed:
        if cfg.device_target == "Ascend":
            context.set_context(device_id=device_id)
            init("hccl")
        else:
            assert cfg.device_target == "GPU"
            init("nccl")
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        device_num = cfg.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        if cfg.device_target in ["Ascend", "GPU"]:
            context.set_context(device_id=device_id)
    config.set_enable_shared_mem(False) # we may get OOM when it set to 'True'
    cfg.logger = get_logger(cfg.save_dir, "R2plus1D", cfg.rank)
    cfg.logger.save_args(cfg)
    cfg.rank_save_ckpt_flag = not (cfg.is_save_on_master and cfg.rank)
    copy_data_from_obs()
    train()
    export_models()
    copy_data_to_obs()
    cfg.logger.info('All task finished!')
