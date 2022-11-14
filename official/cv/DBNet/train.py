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
"""DBNet Train."""
import os
import sys

import mindspore as ms
from mindspore import nn

import src.modules.loss as loss
from src.modules.model import get_dbnet, WithLossCell
from src.utils.callback import DBNetMonitor
from src.utils.learning_rate import warmup_polydecay
from src.utils.env import init_env
from src.datasets.load import create_dataset
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def init_group_params(net, weight_decay):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


@moxing_wrapper()
def train():
    init_env(config)
    train_dataset, steps_pre_epoch = create_dataset(config, True)

    ## Model
    net = get_dbnet(config.net, config, isTrain=True)
    if config.train.pretrained_ckpt:
        ms.load_checkpoint(net, config.train.pretrained_ckpt)
        print("load pretrained checkpoint:", config.train.pretrained_ckpt)

    lr = ms.Tensor(warmup_polydecay(base_lr=config.optimizer.lr.base_lr,
                                    target_lr=config.optimizer.lr.target_lr,
                                    warmup_epoch=config.optimizer.lr.warmup_epoch,
                                    total_epoch=config.train.total_epochs,
                                    start_epoch=config.train.start_epoch_num,
                                    steps_pre_epoch=steps_pre_epoch,
                                    factor=config.optimizer.lr.factor))
    if config.optimizer.type == "sgd":
        print("Use Momentum")
        opt = nn.Momentum(params=init_group_params(net, config.optimizer.weight_decay),
                          learning_rate=lr,
                          momentum=config.optimizer.momentum)
    elif config.optimizer.type == "adam":
        if hasattr(nn.Adam, "use_amsgrad"):
            print("Use amsgrad Adam")
            opt = nn.Adam(net.trainable_params(), learning_rate=lr, use_amsgrad=True)
        else:
            print("Use Adam")
            opt = nn.Adam(net.trainable_params(), learning_rate=lr)
    else:
        raise ValueError(f"Not support optimizer: {config.optimizer.type}")
    # Loss function
    criterion = loss.L1BalanceCELoss(eps=config.loss.eps, l1_scale=config.loss.l1_scale,
                                     bce_scale=config.loss.bce_scale, bce_replace=config.loss.bce_replace)
    if config.mix_precision:
        # only resnet run with float16
        net.to_float(ms.float32)
        net.backbone.to_float(ms.float16)
    net_with_loss = WithLossCell(net, criterion)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss,
                                                 optimizer=opt,
                                                 scale_sense=nn.FixedLossScaleUpdateCell(1024.))
    model = ms.Model(train_net)
    model.train(config.train.total_epochs - config.train.start_epoch_num, train_dataset,
                callbacks=[DBNetMonitor(config, train_net=train_net)],
                dataset_sink_mode=config.train.dataset_sink_mode)

if __name__ == '__main__':
    train()
    print("Train has completed.")
