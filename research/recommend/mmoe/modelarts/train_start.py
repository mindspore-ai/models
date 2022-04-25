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

"""
#################train MMoE example on census-income data########################
python train.py
"""

import os
import datetime
import numpy as np
import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net, Tensor, export
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.common import set_seed

from src.mmoe import TrainStepWrap
from src.model_utils.moxing_adapter import moxing_wrapper
from src.load_dataset import create_dataset
from src.mmoe import MMoE_Layer, MMoE
from src.model_utils.config import config
from src.mmoe import LossForMultiLabel, NetWithLossClass
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
from src.get_lr import get_lr

set_seed(1)


def get_latest_ckpt():
    """get latest ckpt"""
    ckpt_path = config.ckpt_path
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_path) if ckpt_file.endswith(".ckpt")]
    if not ckpt_files:
        return None
    latest_ckpt_file = sorted(ckpt_files)[-1]
    return latest_ckpt_file


def modelarts_process():
    pass


@moxing_wrapper(pre_process=modelarts_process)
def export_mmoe():
    """export MMoE"""
    latest_ckpt_file = get_latest_ckpt()
    if not latest_ckpt_file:
        print("Not found ckpt file")
        return
    config.ckpt_file_path = os.path.join(config.ckpt_path, latest_ckpt_file)
    config.file_name = os.path.join(config.ckpt_path, config.file_name)
    net = MMoE(num_features=config.num_features, num_experts=config.num_experts, units=config.units)
    param_dict = load_checkpoint(config.ckpt_file_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([1, 499]), ms.float16)
    config.file_format = "AIR"
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """train function"""
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())

    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(save_graphs=False)
    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    device_num = get_device_num()

    if config.run_distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif device_target == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())
    print("init finished.")

    config.data_path = config.data_url
    ds_train = create_dataset(config.data_path, config.batch_size, training=True, \
        target=config.device_target, run_distribute=config.run_distribute)

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size.")
    print("create dataset finished.")

    net = MMoE_Layer(input_size=config.num_features, num_experts=config.num_experts, units=config.units)
    print("model created.")
    loss = LossForMultiLabel()
    loss_net = NetWithLossClass(net, loss)

    step_per_size = ds_train.get_dataset_size()
    print("train dataset size:", step_per_size)

    if config.run_distribute:
        learning_rate = get_lr(0.0005, config.epoch_size, step_per_size, step_per_size * 2)
    else:
        learning_rate = get_lr(0.001, config.epoch_size, step_per_size, step_per_size * 5)
    opt = Adam(net.trainable_params(),
               learning_rate=learning_rate,
               beta1=0.9,
               beta2=0.999,
               eps=1e-7,
               weight_decay=0.0,
               loss_scale=1.0)
    scale_update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12,
                                                   scale_factor=2,
                                                   scale_window=1000)
    train_net = TrainStepWrap(loss_net, opt, scale_update_cell)
    train_net.set_train()
    model = Model(train_net)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor(step_per_size)
    config_ck = CheckpointConfig(save_checkpoint_steps=step_per_size, keep_checkpoint_max=100)
    callbacks_list = [time_cb, loss_cb]
    if get_rank_id() == 0:
        config.ckpt_path = config.train_url
        config.ckpt_path = os.path.join(config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        ckpoint_cb = ModelCheckpoint(prefix='MMoE_train', directory=config.ckpt_path, config=config_ck)
        callbacks_list.append(ckpoint_cb)

    print("train start!")
    model.train(epoch=config.epoch_size,
                train_dataset=ds_train,
                callbacks=callbacks_list,
                dataset_sink_mode=config.dataset_sink_mode)


if __name__ == '__main__':
    run_train()
    export_mmoe()
