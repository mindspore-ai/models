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
''' binary library'''
import numpy as np
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter, ops

Signer = ops.Sign()

class BinaryActivation(nn.Cell):
    '''
        Binarize activation with adaptive binary set
    '''
    def __init__(self):
        super(BinaryActivation, self).__init__()
        self.alpha_a = Parameter(Tensor(1., dtype=mstype.float32), name="alpha_a", requires_grad=True)
        self.beta_a = Parameter(Tensor(0., dtype=mstype.float32), name="beta_a", requires_grad=True)

        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def construct(self, x):
        x_norm = (x - self.beta_a) / self.alpha_a
        # clip range
        x_norm = self.hardtanh(x_norm)
        x_bin = Signer(x_norm)
        x_adabin = (x_bin + self.beta_a)*self.alpha_a
        return x_adabin

def BinaryWeight(weight):
    '''
        Binarize activation with adaptive binary set
    '''
    beta_w = weight.mean((1, 2, 3)).view(-1, 1, 1, 1)
    alpha_w = weight.std((1, 2, 3)).view(-1, 1, 1, 1)

    w_norm = (weight - beta_w) / alpha_w
    w_bin = Signer(w_norm)
    w_adabin = w_bin * alpha_w + beta_w
    return w_adabin

class AdaBinConv2d(nn.Conv2d):
    '''
        AdaBin Binary Neural Network
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, \
                 group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW", \
                 a_bit=1, w_bit=1):
        super(AdaBinConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, \
                                            dilation, group, has_bias, weight_init, bias_init, data_format)
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.binary_a = BinaryActivation()
        self.conv2d = P.Conv2D(out_channel=out_channels,
                               kernel_size=kernel_size,
                               mode=1,
                               pad_mode=pad_mode,
                               pad=padding,
                               stride=stride,
                               dilation=dilation,
                               group=group)

    def construct(self, x):
        if self.a_bit == 1:
            x = self.binary_a(x)

        if self.w_bit == 1:
            weight = BinaryWeight(self.weight)
        else:
            weight = self.weight

        output = self.conv2d(x, weight)

        return output

class Maxout(nn.Cell):
    '''
        Nonlinear function
    '''
    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super(Maxout, self).__init__()
        self.neg_scale = Parameter(Tensor(neg_init*np.ones((1, channel, 1, 1)), \
                                   dtype=mstype.float32), name="neg_scale", requires_grad=True)
        self.pos_scale = Parameter(Tensor(pos_init*np.ones((1, channel, 1, 1)), \
                                   dtype=mstype.float32), name="pos_scale", requires_grad=True)
        self.relu = nn.ReLU()

    def construct(self, x):
        ''' Maxout '''
        x = self.pos_scale*self.relu(x) - self.neg_scale*self.relu(-x)
        return x
