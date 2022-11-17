#!/bin/bash
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
import argparse
from collections import namedtuple
import warnings
from tqdm import tqdm
import gin

import mindspore
from mindspore import context
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net

from src import ds_load
from src.cnv_for_train import TrainOneStepCell, MyWithLossCell
from src.cnv_model import OrigamiNet
from src.cnv_model import ginM

warnings.filterwarnings('ignore')


parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)

pO = None
OnceExecWorker = None


parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=6)


def gInit(opt):
    global pO, OnceExecWorker
    gin.parse_config_file(opt.gin)
    pO = parOptions(**{ginM('dist'): True})

    OnceExecWorker = True



@gin.configurable
def train(opt, AMP, WdB, train_data_path, train_data_list, test_data_path, test_data_list,
          train_batch_size, val_batch_size, lr, valInterval, save_model_path, model_prefix, continue_model=''):

    train_dataset_r = ds_load.myLoadDS(train_data_list, train_data_path)

    train_dataset_s = ds.GeneratorDataset(source=train_dataset_r,
                                          num_parallel_workers=16, column_names=["data", "label"])
    train_dataset_s = train_dataset_s.shuffle(train_dataset_r.get_dataset_size())


    train_dataset_s = train_dataset_s.batch(batch_size=train_batch_size)
    batch_num = train_dataset_s.get_dataset_size()

    train_loader = train_dataset_s.create_tuple_iterator()

    network = OrigamiNet()
    print("continue model: ", continue_model)
    param_dict = load_checkpoint(continue_model)
    load_param_into_net(network, param_dict)
    print("network parameters init")
    lr_scheduler = nn.exponential_decay_lr(learning_rate=lr, decay_rate=10 ** (-1/90000),
                                           decay_epoch=1, step_per_epoch=batch_num, total_step=batch_num * valInterval)
    optimzer = nn.Adam(network.trainable_params(), learning_rate=lr_scheduler)

    loss_fn = ops.CTCLoss()
    net = MyWithLossCell(network, loss_fn)
    net = TrainOneStepCell(net, optimzer)
    print("Trainonestep model init")

    for epoch in tqdm(range(valInterval)):
        loss = 0
        print("epoch")
        for index, data in enumerate(train_loader):
            img = data[0]
            label = data[1][0]
            out, _ = net(img, label)
            loss += float(out[0][0].asnumpy())
            print("epoch: {0}/{1}, step: {2}, losses: {3}".format(epoch + 1,
                                                                  valInterval, index, out[0][0].asnumpy(), flush=True))


        loss_e = loss / 100
        print("--------------epoch: {0}/{1}, losses: {2}-------------".format(epoch + 1,
                                                                              valInterval, loss_e, flush=True))

        if (epoch + 1) % 10 == 0:
            name = model_prefix + str(epoch+1) + ".ckpt"
            path = os.path.join(save_model_path, name)
            mindspore.save_checkpoint(net.network.backbone_network(), path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', help='Gin config file', default="parameters/hwdb.gin")

    opt_out = parser.parse_args()
    gInit(opt_out)
    opt_out.manualSeed = ginM('manualSeed')
    opt_out.port = ginM('port')

    print("opt.num_gpu: ", opt_out.num_gpu)
    print(opt_out)
    train(opt_out)
