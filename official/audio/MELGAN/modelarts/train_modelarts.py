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
"""MelGAN train"""
import os
import time

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.dataset as de
import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import RunContext, ModelCheckpoint, CheckpointConfig, _InternalCallbackParam
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.dataset import Generator1D
from src.loss import MelganLoss_G, MelganLoss_D
from src.model import MultiDiscriminator, Generator
from src.model_utils.config import config as cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.sampler import DistributedSampler
from src.trainonestep import TrainOneStepCellGEN, TrainOneStepCellDIS

set_seed(1)


class BuildGenNetwork(nn.Cell):
    """build generator"""

    def __init__(self, network, criterion):
        super(BuildGenNetwork, self).__init__(auto_prefix=False)
        self.network = network
        self.criterion = criterion

    def construct(self, data):
        fake_wav = self.network(data)
        return fake_wav


class BuildDisNetwork(nn.Cell):
    """build discriminator"""

    def __init__(self, network, criterion):
        super(BuildDisNetwork, self).__init__(auto_prefix=False)
        self.network = network
        self.criterion = criterion

    def construct(self, fake_wav, wav):
        y1 = self.network(fake_wav)
        y2 = self.network(wav)
        loss = self.criterion(y1, y2)
        return loss


@moxing_wrapper()
def train():
    """main train process"""
    # init distributed
    if cfg.run_distribute:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=8,
                                          parameter_broadcast=True)
    else:
        cfg.rank = 0
        cfg.group_size = 1
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=cfg.device_id)
    # get network and init
    net_D = MultiDiscriminator()
    net_G = Generator(alpha=cfg.leaky_alpha)

    criterion_G = MelganLoss_G()
    criterion_D = MelganLoss_D()

    gen_network_train = BuildGenNetwork(net_G, criterion_G)
    gen_network_train.set_train()
    dis_network_train_1 = BuildDisNetwork(net_D, criterion_G)
    dis_network_train_1.set_train()
    dis_network_train_2 = BuildDisNetwork(net_D, criterion_D)
    dis_network_train_2.set_train()
    scale_manager = DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_factor=2, scale_window=2000)

    # optimizer
    opt_G = nn.Adam(params=net_G.trainable_params(), learning_rate=cfg.lr_g, beta1=cfg.beta1, beta2=cfg.beta2,
                    weight_decay=cfg.weight_decay)
    opt_D = nn.Adam(params=net_D.trainable_params(), learning_rate=cfg.lr_d, beta1=cfg.beta1, beta2=cfg.beta2,
                    weight_decay=cfg.weight_decay)
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(net_G, param_dict)
        load_param_into_net(net_D, param_dict)

    gen_network_train_wrap = TrainOneStepCellGEN(gen_network_train, opt_G, dis_network_train_1, criterion_G)
    dis_network_train_wrap = TrainOneStepCellDIS(gen_network_train, dis_network_train_2, opt_D, criterion_D)

    # dataloader
    Wavmeldataset = Generator1D(cfg.data_path, cfg.train_length, cfg.hop_size)
    distributed_sampler = DistributedSampler(len(Wavmeldataset), cfg.group_size, cfg.rank, shuffle=True)
    dataset = de.GeneratorDataset(Wavmeldataset, ["data", "wav", "datad", "wavd"], sampler=distributed_sampler)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=True)

    # checkpoint save
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_steps, keep_checkpoint_max=100000)
    ckpt_cb = ModelCheckpoint(prefix=cfg.save_checkpoint_name, directory=cfg.train_url, config=config_ck)
    cb_params = _InternalCallbackParam()
    cb_params.train_network = gen_network_train_wrap
    cb_params.epoch_num = cfg.epoch_size
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)

    i = 1
    print(cfg.epoch_size)
    epoch_t = time.perf_counter()

    # epoch loop
    for epoch in range(cfg.epoch_size):
        cb_params.cur_epoch_num = epoch + 1
        for data, wav, datad, wavd in dataset.create_tuple_iterator():
            scaling_sens = Tensor(scale_manager.get_loss_scale(), dtype=mstype.float32)
            start = time.perf_counter()
            data = (data + 5.0) / 5.0
            datad = (datad + 5.0) / 5.0

            _, loss_G, cond_g = gen_network_train_wrap(Tensor(wav, mstype.float32), Tensor(data, mstype.float32),
                                                       scaling_sens)

            _, loss_D, cond_d = dis_network_train_wrap(Tensor(datad, mstype.float32), Tensor(wavd, mstype.float32),
                                                       scaling_sens)
            if cond_g:
                scale_manager.update_loss_scale(cond_g)
            else:
                scale_manager.update_loss_scale(False)
            if cond_d:
                scale_manager.update_loss_scale(cond_d)
            else:
                scale_manager.update_loss_scale(False)
            duration = time.perf_counter() - start

            print(
                '{}epoch {}iter loss_G={} loss_D={} {:.2f}s/it'.format(epoch + 1, i, loss_G.asnumpy(), loss_D.asnumpy(),
                                                                       duration))

            i = i + 1
            if cfg.rank == 0:
                cb_params.cur_step_num = i
                cb_params.batch_num = i
                ckpt_cb.step_end(run_context)

    duration = time.perf_counter() - epoch_t
    print('finish in {:.2f}mins'.format(duration / 60))

    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[1, 80, 240]), ms.float32)
    export(net_G, input_arr, file_name=os.path.join(cfg.train_url, 'melgan_final'), file_format="AIR")


if __name__ == "__main__":
    train()
