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
"""train enet"""
import os

import moxing as mox
import numpy as np

from mindspore import Model, context, nn, load_param_into_net, export
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import _update_param, load_checkpoint

from src.config import (TrainConfig_1, TrainConfig_2, TrainConfig_3,
                        ms_train_data, num_class, repeat, run_distribute, save_path, weight_init)
from src.criterion import SoftmaxCrossEntropyLoss
from src.dataset import getCityScapesDataLoader_mindrecordDataset
from src.model import Encoder_pred, Enet
from src.util import getCityLossWeight


def attach(enet, encoder_pretrain):
    """move the params in encoder to enet"""
    print("attach decoder.")
    encoder_trained_par = encoder_pretrain.parameters_dict()
    enet_par = enet.parameters_dict()
    for name, param_old in encoder_trained_par.items():
        if name.startswith("encoder"):
            _update_param(enet_par[name], param_old, False)


def train(ckpt_path_, trainConfig_, rank_id, rank_size, stage_):
    """train enet"""
    print("stage:", stage_)
    save_prefix = "Encoder" if trainConfig_.encode else "ENet"
    if trainConfig_.epoch == 0:
        raise RuntimeError("epoch num cannot be zero")

    if trainConfig_.encode:
        network = Encoder_pred(num_class, weight_init)
    else:
        network = Enet(num_class, weight_init)
    if not os.path.exists(ckpt_path_):
        print("load no ckpt file.")
    else:
        load_checkpoint(ckpt_file_name=ckpt_path_, net=network)
        print("load ckpt file:", ckpt_path_)

    # attach decoder
    if trainConfig_.attach_decoder:
        network_enet = Enet(num_class, weight_init)
        attach(network_enet, network)
        network = network_enet
    dataloader = getCityScapesDataLoader_mindrecordDataset(stage_, ms_train_data, 6,
                                                           trainConfig_.encode, trainConfig_.train_img_size,
                                                           shuffle=True, aug=True,
                                                           rank_id=rank_id, global_size=rank_size, repeat=repeat)
    opt = nn.Adam(network.trainable_params(), trainConfig_.lr,
                  weight_decay=1e-4, eps=1e-08)
    loss = SoftmaxCrossEntropyLoss(num_class, getCityLossWeight(trainConfig_.encode))

    loss_scale_manager = DynamicLossScaleManager()
    wrapper = Model(network, loss, opt, loss_scale_manager=loss_scale_manager,
                    keep_batchnorm_fp32=True)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor()

    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=
                                     trainConfig_.epoch_num_save * dataloader.get_dataset_size(), \
                                     keep_checkpoint_max=9999)
        saveModel_cb = ModelCheckpoint(prefix=save_prefix, directory= \
            "./", config=config_ck)
        call_backs = [saveModel_cb, time_cb, loss_cb]
    else:
        call_backs = [time_cb, loss_cb]

    print("============== Starting {} Training ==============".format(save_prefix))
    wrapper.train(trainConfig_.epoch, dataloader, callbacks=call_backs, dataset_sink_mode=True)
    return network


def export_models(ckptfile):
    print("exporting model....")
    net = Enet(20, "XavierUniform", train=False)
    param_dict = load_checkpoint(ckptfile)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([1, 3, 512, 1024]).astype(np.float32))
    export(net, input_arr, file_name="ENet.air", file_format="AIR")
    print("export model finished....")


if __name__ == "__main__":
    rank_id_ = 0
    rank_size_ = 1
    if run_distribute:
        context.set_auto_parallel_context(parameter_broadcast=True)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=False)
        init()
        rank_id_ = get_rank()
        rank_size_ = get_group_size()
    trainConfig = {
        1: TrainConfig_1(),
        2: TrainConfig_2(),
        3: TrainConfig_3()
    }
    for i in [1, 2, 3]:
        data_loader = getCityScapesDataLoader_mindrecordDataset(i, ms_train_data, 6,
                                                                trainConfig[i].encode, trainConfig[i].train_img_size,
                                                                shuffle=True, aug=True,
                                                                rank_id=rank_id_, global_size=rank_size_, repeat=repeat)
        steps = int(TrainConfig_1().epoch_num_save * data_loader.get_dataset_size() / 5)
        if i == 1:
            ckpt_path = ""
        elif i == 2:
            ckpt_path = "./Encoder-{}_{}.ckpt".format(TrainConfig_1().epoch, steps)
        else:
            ckpt_path = "./Encoder_1-{}_{}.ckpt".format(TrainConfig_2().epoch, steps)
        network_ = train(ckpt_path, trainConfig[i], rank_id=rank_id_,
                         rank_size=rank_size_, stage_=i)
    ckpt = "./ENet-{}_{}.ckpt".format(TrainConfig_3().epoch, steps)
    export_models(ckpt)
    mox.file.copy_parallel("./", save_path)
