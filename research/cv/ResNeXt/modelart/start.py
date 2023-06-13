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
"""train ImageNet."""
import os
import time
import datetime
import numpy as np
import moxing as mox

import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig, Callback
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.common import set_seed
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
import mindspore.common.initializer as weight_init
from mindspore.common import dtype as mstype

from src.dataset import classification_dataset
from src.crossentropy import CrossEntropy
from src.lr_generator import get_lr
from src.utils.logging import get_logger
from src.utils.optimizers__init__ import get_param_groups
from src.image_classification import get_network
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.utils.auto_mixed_precision import auto_mixed_precision

set_seed(1)


class BuildTrainNetwork(nn.Cell):
    """build training network"""

    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


class ProgressMonitor(Callback):
    """monitor loss and time"""

    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args
        self.ckpt_history = []

    def begin(self, run_context):
        self.args.logger.info("start network train...")

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = (
            self.args.per_batch_size * (me_step - self.me_epoch_start_step_num) * self.args.group_size / time_used
        )
        self.args.logger.info(
            "epoch[{}], iter[{}], loss:{}, mean_fps:{:.2f}"
            "imgs/sec".format(real_epoch, me_step, cb_params.net_outputs, fps_mean)
        )

        if self.args.rank_save_ckpt_flag:
            import glob

            ckpts = glob.glob(os.path.join(self.args.outputs_dir, "*.ckpt"))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith("{}-".format(self.args.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.args.logger.info(
                    "epoch[{}], iter[{}], loss:{}, ckpt:{},"
                    "ckpt_fn:{}".format(real_epoch, me_step, cb_params.net_outputs, ckpt, ckpt_fn)
                )

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info("end network train...")


def set_parameters():
    """parameters"""
    context.set_context(
        mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, device_target=config.device_target, save_graphs=False
    )
    # init distributed
    if config.run_distribute:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()
    else:
        config.rank = 0
        config.group_size = 1
        init()

    if config.is_dynamic_loss_scale == 1:
        config.loss_scale = 1  # for dynamic loss scale can not set loss scale in momentum opt

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.output_path, datetime.datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S"))
    config.logger = get_logger(config.outputs_dir, config.rank)
    return config


def set_graph_kernel_context(device_target):
    if device_target == "GPU":
        context.set_context(enable_graph_kernel=True)


def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir) if ckpt_file.endswith(".ckpt")]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def run_export(ckpt_dir):
    """run export."""
    checkpoint_file_path = _get_last_ckpt(ckpt_dir)
    network = get_network(network=config.network, num_classes=config.num_classes, platform=config.device_target)

    param_dict = load_checkpoint(checkpoint_file_path)
    load_param_into_net(network, param_dict)
    if config.device_target == "Ascend":
        network.to_float(mstype.float16)
    else:
        auto_mixed_precision(network)
    network.set_train(False)
    input_shp = [config.batch_size, 3, config.height, config.width]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(network, input_array, file_name=config.file_name, file_format=config.file_format)
    mox.file.copy_parallel(os.getcwd(), config.train_url)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def init_weight(net):
    if os.path.exists(config.checkpoint_file_path):
        param_dict = load_checkpoint(config.checkpoint_file_path)
        filter_weight = True
        print(1111111)
        if filter_weight:
            filter_list = ["head.fc.weight", "head.fc.bias"]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)

    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    weight_init.initializer(weight_init.XavierUniform(), cell.weight.shape, cell.weight.dtype)
                )
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    weight_init.initializer(weight_init.TruncatedNormal(), cell.weight.shape, cell.weight.dtype)
                )


@moxing_wrapper()
def train():
    """training process"""
    set_parameters()
    if os.getenv("DEVICE_ID", "not_set").isdigit():
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))
    set_graph_kernel_context(config.device_target)

    # init distributed
    if config.run_distribute:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(
            parallel_mode=parallel_mode, device_num=config.group_size, gradients_mean=True
        )
    # dataloader
    de_dataset = classification_dataset(
        config.data_path,
        config.image_size,
        config.per_batch_size,
        1,
        config.rank,
        config.group_size,
        num_parallel_workers=8,
    )
    config.steps_per_epoch = de_dataset.get_dataset_size()

    config.logger.save_args(config)

    # network
    config.logger.important_info("start create network")
    # get network and init
    network = get_network(network=config.network, num_classes=config.num_classes, platform=config.device_target)
    init_weight(network)

    # lr scheduler
    lr = get_lr(config)

    # optimizer
    opt = Momentum(
        params=get_param_groups(network),
        learning_rate=Tensor(lr),
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        loss_scale=config.loss_scale,
    )

    # loss
    if not config.label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)

    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(
        network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager, metrics={"acc"}, amp_level="O3"
    )

    # checkpoint save
    progress_cb = ProgressMonitor(config)
    callbacks = [
        progress_cb,
    ]
    if config.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=config.ckpt_interval * config.steps_per_epoch,
            keep_checkpoint_max=config.ckpt_save_max,
        )
        save_ckpt_path = os.path.join(config.outputs_dir, "ckpt_" + str(config.rank) + "/")
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=save_ckpt_path, prefix="{}".format(config.rank))
        callbacks.append(ckpt_cb)

    model.train(config.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=True)
    run_export(save_ckpt_path)


if __name__ == "__main__":
    train()
