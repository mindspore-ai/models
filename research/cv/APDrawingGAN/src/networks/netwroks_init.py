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
"""
initialize network
"""

from mindspore.common import initializer
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np


def init_weights(net, init_type='normal', gain=0.02):
    """
    initialize weights
    """
    net.init_parameters_data()
    for _, m in net.cells_and_names():
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                weight = initializer.initializer(initializer.Normal(sigma=gain),
                                                 shape=m.weight.data.shape, dtype=mstype.float32)
                m.weight.set_data(weight)
            elif init_type == 'xavier':
                weight = initializer.initializer(initializer.XavierUniform(gain=gain),
                                                 shape=m.weight.data.shape, dtype=mstype.float32)
                m.weight.set_data(weight)
            elif init_type == 'kaiming':
                weight = initializer.initializer(initializer.HeNormal(mode='fan_in', negative_slope=0),
                                                 shape=m.weight.data.shape, dtype=mstype.float32)
                m.weight.set_data(weight)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

        elif classname.find('BatchNorm2d') != -1:
            m.gamma.set_data(
                Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
            m.beta.set_data(
                Tensor(np.zeros(m.beta.data.shape, dtype="float32")))

def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net
