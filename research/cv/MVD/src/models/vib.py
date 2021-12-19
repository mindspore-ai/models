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
""" vib.py """

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore.common.initializer import Initializer, random_normal
from src.utils import assignment


def weights_init_kaiming(module):
    """
    weight init
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_in'),
                                                module.weight.shape, module.weight.dtype))
    elif classname.find('Linear') != -1:
        module.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_out'),
                                                module.weight.shape, module.weight.dtype))
        module.bias.set_data(init.initializer(init.Zero(), module.bias.shape, module.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        module.gamma.set_data(init.initializer(NormalWithMean(mu=1, sigma=0.01),
                                               module.gamma.shape, module.gamma.dtype))
        module.beta.set_data(init.initializer(init.Zero(), module.beta.shape, module.beta.dtype))


def weights_init_classifier(module):
    """
    weight init
    """
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        module.gamma.set_data(init.initializer(init.Normal(sigma=0.001),
                                               module.gamma.shape, module.gamma.dtype))
        if module.bias:
            module.bias.set_data(init.initializer(init.Zero(),
                                                  module.bias.shape, module.bias.dtype))


class NormalWithMean(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, normal array.
    """

    def __init__(self, mu=0, sigma=0.01):
        super(NormalWithMean, self).__init__(sigma=sigma)
        self.miu = mu
        self.sigma = sigma

    def _initialize(self, arr):
        """
        init
        """
        seed, seed2 = self.seed
        output_tensor = ms.Tensor(np.zeros(arr.shape, dtype=np.float32) +
                                  np.ones(arr.shape, dtype=np.float32) * self.miu)
        random_normal(arr.shape, seed, seed2, output_tensor)
        output_data = output_tensor.asnumpy()
        output_data *= self.sigma
        assignment(arr, output_data)


########################################################################
# Variational Distillation
########################################################################
class ChannelCompress(nn.Cell):
    """
    ChannelCompress
    """

    def __init__(self, in_ch=2048, out_ch=256):
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Dense(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_features=num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Dense(num_bottleneck, num_bottleneck // 2)]
        add_block += [nn.BatchNorm1d(num_bottleneck // 2)]
        add_block += [nn.ReLU()]

        add_block += [nn.Dense(num_bottleneck // 2, out_ch)]

        add_block = nn.SequentialCell(add_block)

        weights_init_kaiming(add_block)

        self.model = add_block

    def construct(self, x):
        """
        construct
        """
        x = self.model(x)
        return x


########################################################################
# Variational Distillation
########################################################################
class VIB(nn.Cell):
    """
    VIB module
    """

    def __init__(self, in_ch=2048, z_dim=256, num_class=395):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB
        classifier = []
        classifier += [nn.Dense(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Dense(self.out_ch // 2, self.num_class)]
        classifier = nn.SequentialCell(classifier)
        weights_init_classifier(classifier)
        self.classifier = classifier

    def construct(self, v):
        """
        construct
        """

        z_given_v = self.bottleneck(v)
        logits_given_z = self.classifier(z_given_v)
        return z_given_v, logits_given_z
