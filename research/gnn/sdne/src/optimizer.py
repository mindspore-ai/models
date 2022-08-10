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
"""
python optimizer.py
"""
import numpy as np
from mindspore import Tensor
from mindspore import nn


def lr_generator(lr_init, total_epochs, steps_per_epoch, schedule, k):
    """
    return a learning rate Tensor
    """
    lr_each_step = []
    for i in range(total_epochs):
        if i in schedule:
            lr_init *= k
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return Tensor(lr_each_step)

def optimizer(net, cfg, total_epochs, steps_per_epoch):
    """
    return a optimizer
    """
    learning_rate = cfg['learning_rate'] if 'learning_rate' in cfg else 0.1
    if 'schedule' in cfg:
        learning_rate = lr_generator(learning_rate, total_epochs, steps_per_epoch, cfg['schedule'], cfg['k'])
    weight_decay = cfg['weight_decay'] if 'weight_decay' in cfg else 0.0001
    optim = None
    if cfg['name'] == 'SGD':
        optim = nn.SGD(params=net.trainable_params(), learning_rate=learning_rate,
                       weight_decay=weight_decay)
    elif cfg['name'] == 'RMSProp':
        optim = nn.RMSProp(params=net.trainable_params(), learning_rate=learning_rate,
                           weight_decay=weight_decay)
    elif cfg['name'] == 'Adam':
        optim = nn.Adam(params=net.trainable_params(), learning_rate=learning_rate,
                        weight_decay=weight_decay)
    else:
        print('unsupported optimizer.')

    return optim
