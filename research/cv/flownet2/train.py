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
import os
import datetime
import glob
import mindspore as ms
import mindspore.dataset as ds
import mindspore.log as logger
import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore.nn.optim.adam import Adam
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed

from src.eval_callback import EvalCallBack
import src.dataset as datasets
import src.models as models
from src.metric import FlowNetEPE
import src.model_utils.tools as tools
from src.model_utils.config import config


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = config.save_checkpoint_path
    if config.run_distribute:
        ckpt_save_dir = ckpt_save_dir + "/ckpt_" + str(get_rank()) + "/"
    return ckpt_save_dir

def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds, dataset_sink_mode=False)
    return res[metrics_name]


def load_pre_trained_checkpoint(net, pre_trained, checkpoint_path):
    param_dict = None
    if pre_trained:
        if os.path.isdir(checkpoint_path):
            ckpt_save_dir = os.path.join(checkpoint_path, "ckpt_0")
            ckpt_pattern = os.path.join(ckpt_save_dir, "*.ckpt")
            ckpt_files = glob.glob(ckpt_pattern)
            if not ckpt_files:
                logger.warning(f"There is no ckpt file in {ckpt_save_dir}, "
                               f"pre_trained is unsupported.")
            else:
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                time_stamp = datetime.datetime.now()
                print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')}"
                      f" pre trained ckpt model {ckpt_files[0]} loading",
                      flush=True)
                param_dict = load_checkpoint(ckpt_files[0])
        elif os.path.isfile(checkpoint_path):
            param_dict = load_checkpoint(checkpoint_path)
        else:
            print(f"Invalid pre_trained {checkpoint_path} parameter.")
            return
        load_param_into_net(net, param_dict)
        print(f"loaded param from {checkpoint_path} into net")


def add_ckpt_callback(step_size, ckpt_save_dir, cbs):
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size * config.save_ckpt_interval,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="flownet2_", directory=ckpt_save_dir, config=config_ck)
        cbs += [ckpoint_cb]


def add_eval_callback(model, ckpt_save_dir, cbs):
    if config.run_evalCallback:
        if config.eval_data_path is None or (not os.path.isdir(config.eval_data_path)):
            raise ValueError("{} is not a existing path.".format(config.eval_data_path))

        config.eval_dataset_class = tools.module_to_dict(datasets)[config.eval_data]
        flownet_eval_gen = config.eval_dataset_class("Center", config.crop_size, config.eval_size,
                                                     config.eval_data_path)
        eval_dataset = ds.GeneratorDataset(flownet_eval_gen, ["images", "flow"],
                                           num_parallel_workers=config.num_parallel_workers,
                                           max_rowsize=config.max_rowsize)
        eval_dataset = eval_dataset.batch(config.batch_size)

        eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "FlowNetEPE"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=config.save_best_ckpt,
                               ckpt_directory=ckpt_save_dir, besk_ckpt_name="best_acc.ckpt",
                               metrics_name="FlowNetEPE")
        cbs += [eval_cb]


def run_train():
    set_seed(config.seed)
    ms.set_context(mode=ms.context.GRAPH_MODE, enable_graph_kernel=True, device_target=config.device_target)
    ds.config.set_enable_shared_mem(False)
    if config.device_target == "GPU":
        if config.run_distribute:
            init()
            parallel_mode = ParallelMode.DATA_PARALLEL
            rank = get_rank()
            group_size = get_group_size()
        else:
            parallel_mode = ParallelMode.STAND_ALONE
            rank = 0
            group_size = 1

        ms.context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=group_size)

    # load dataset by config param
    config.training_dataset_class = tools.module_to_dict(datasets)[config.train_data]
    flownet_train_gen = config.training_dataset_class(config.crop_type, config.crop_size, config.eval_size,
                                                      config.train_data_path)
    sampler = datasets.DistributedSampler(flownet_train_gen, rank, group_size, shuffle=True)
    train_dataset = ds.GeneratorDataset(flownet_train_gen, ["images", "flow"],
                                        sampler=sampler, num_parallel_workers=config.num_parallel_workers)
    train_dataset = train_dataset.batch(config.batch_size)
    step_size = train_dataset.get_dataset_size()

    # load model by config param
    config.model_class = tools.module_to_dict(models)[config.model]
    net = config.model_class(config.rgb_max, config.batchNorm)

    loss = nn.L1Loss()
    if config.is_dynamicLoss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.scale, drop_overflow_update=False)

    optim = Adam(params=net.trainable_params(), learning_rate=config.lr)

    load_pre_trained_checkpoint(net, config.pre_trained, config.pre_trained_ckpt_path)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics={'FlowNetEPE': FlowNetEPE()},
                  amp_level="O0", keep_batchnorm_fp32=True, loss_scale_manager=loss_scale_manager)
    # add callback
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]
    ckpt_save_dir = set_save_ckpt_dir()
    add_ckpt_callback(step_size, ckpt_save_dir, cbs)
    add_eval_callback(model, ckpt_save_dir, cbs)

    model.train(config.epoch_size, train_dataset, callbacks=cbs, dataset_sink_mode=True)

if __name__ == '__main__':
    run_train()
