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
'''train'''
import os
import datetime

import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.dataset import config
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size

from src.logger import get_logger
from src.dataset import create_Dataset
from src.models import UNet3Plus, UNet3PlusWithLossCell
from src.util import DiceMetric, EvalCallBack, TempLoss
from src.config import config as cfg

def copy_data_from_obs():
    '''copy_data_from_obs'''
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying train data from obs to cache....")
        mox.file.copy_parallel(cfg.train_data_path, 'cache/dataset')
        cfg.logger.info("copying traindata finished....")
        cfg.train_data_path = 'cache/dataset/'

        if cfg.resume_path:
            cfg.logger.info("copying resume checkpoint from obs to cache....")
            mox.file.copy_parallel(cfg.resume_path, 'cache/resume_path')
            cfg.logger.info("copying resume checkpoint finished....")
            cfg.resume_path = 'cache/resume_path/'

        if cfg.eval_while_train:
            cfg.logger.info("copying val data from obs to cache....")
            mox.file.copy_parallel(cfg.val_data_path, 'cache/vatdataset')
            cfg.logger.info("copying val data finished....")
            cfg.val_data_path = 'cache/vatdataset/'

def copy_data_to_obs():
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(cfg.save_dir, cfg.outer_path)
        cfg.logger.info("copying finished....")

def train():
    '''trian'''
    if cfg.is_distributed:
        assert cfg.device_target == "Ascend"
        init()
        context.set_context(device_id=device_id)

        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        device_num = cfg.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        if cfg.device_target in ["Ascend", "GPU"]:
            context.set_context(device_id=device_id)
    config.set_enable_shared_mem(False)

    train_dataset, cfg.steps_per_epoch = create_Dataset(cfg.train_data_path, cfg.aug, cfg.batch_size,\
                                        cfg.group_size, cfg.rank, shuffle=True)
    f_model = UNet3Plus()

    if cfg.resume_path:
        cfg.resume_path = os.path.join(cfg.resume_path, cfg.resume_name)
        cfg.logger.info('loading resume checkpoint %s into network', str(cfg.resume_path))
        load_param_into_net(f_model, load_checkpoint(cfg.resume_path))
        cfg.logger.info('loaded resume checkpoint %s into network', str(cfg.resume_path))

    optimizer = nn.Adam(params=f_model.trainable_params(), learning_rate=float(cfg.lr))

    time_cb = TimeMonitor(data_size=cfg.steps_per_epoch)
    loss_cb = LossMonitor(50)
    callbacks = [time_cb, loss_cb]

    if cfg.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.steps_per_epoch*cfg.save_every,
                                       keep_checkpoint_max=cfg.ckpt_save_max)
        save_ckpt_path = os.path.join(cfg.save_dir, 'ckpt_' + str(cfg.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='rank_'+str(cfg.rank))
        callbacks.append(ckpt_cb)
    if cfg.eval_while_train == 1:
        loss_f = TempLoss()
        val_dataset, _ = create_Dataset(cfg.val_data_path, 0, cfg.batch_size,\
                                    1, 0, shuffle=False)
        network_eval = Model(f_model, loss_fn=loss_f, metrics={"DiceMetric": DiceMetric()})
        eval_cb = EvalCallBack(network_eval, val_dataset, interval=cfg.eval_steps,
                               eval_start_epoch=cfg.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=cfg.save_dir, besk_ckpt_name=str(cfg.rank)+"_best_map.ckpt")
        callbacks.append(eval_cb)

    model = UNet3PlusWithLossCell(f_model)
    model.set_train()
    model = nn.TrainOneStepCell(model, optimizer)
    model = Model(model)
    model.train(cfg.epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    cfg.logger.info("training finished....")

if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.join(cfg.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    if not cfg.use_modelarts and not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, save_graphs=False)

    cfg.logger = get_logger(cfg.save_dir, "UNet3Plus", cfg.rank)
    cfg.logger.save_args(cfg)
    # select for master rank save ckpt or all rank save, compatible for model parallel
    cfg.rank_save_ckpt_flag = not (cfg.is_save_on_master and cfg.rank)

    copy_data_from_obs()
    train()
    copy_data_to_obs()
    cfg.logger.info('All task finished!')
