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
"""srcnn training"""

import os

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_train_dataset
from src.srcnn import SRCNN

from src.model_utils.config import config
from src.model_utils.moxing_adapter import sync_data


set_seed(1)

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def run_train():
    cfg = config
    data_path = cfg.data_path
    pretrained_ckpt_path = cfg.pre_trained_path
    if cfg.enable_modelarts == "True":
        sync_data(cfg.data_url, data_path)
        if cfg.pre_trained_path:
            sync_data(cfg.pre_trained_path, pretrained_ckpt_path)
    data_path += "/srcnn.mindrecord00"
    output_path = cfg.output_path
    if cfg.device_target == "GPU":
        if cfg.run_distribute:
            init()
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=cfg.device_target,
                            save_graphs=False)
    elif cfg.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=cfg.device_target,
                            device_id=int(os.environ["DEVICE_ID"]),
                            save_graphs=False)
        if cfg.run_distribute:
            init()
    else:
        raise ValueError("Unsupported device target.")

    rank = 0
    device_num = 1
    if cfg.run_distribute:
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    train_dataset = create_train_dataset(data_path, batch_size=cfg.batch_size,
                                         shard_id=rank, num_shard=device_num)

    step_size = train_dataset.get_dataset_size()

    # define net
    net = SRCNN()

    # init weight
    if cfg.pre_trained_path:
        param_dict = load_checkpoint(pretrained_ckpt_path)
        if cfg.filter_weight:
            filter_list = [x.name for x in net.end_point.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(net, param_dict)

    lr = Tensor(float(cfg.lr), ms.float32)

    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)
    loss = nn.MSELoss(reduction='mean')
    model = Model(net, loss_fn=loss, optimizer=opt)

    # define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    if cfg.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        save_ckpt_path = os.path.join(output_path, 'ckpt_' + str(rank) + '/')
        ckpt_cb = ModelCheckpoint(prefix="srcnn", directory=save_ckpt_path, config=config_ck)
        callbacks.append(ckpt_cb)

    model.train(cfg.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    if cfg.enable_modelarts == "True":
        sync_data(output_path, cfg.train_url)

if __name__ == '__main__':
    run_train()
