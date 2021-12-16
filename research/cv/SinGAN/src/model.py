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
"""SinGAN Network Topology"""

import math
from mindspore import nn
from mindspore.common import initializer as init
from src.block import Conv2dBlock
class Gen(nn.Cell):
    """Generator"""
    def __init__(self, opt):
        super().__init__()
        nfc = opt.nfc
        num_layer = opt.num_layer
        ker_size = opt.ker_size
        stride = opt.stride
        padd_size = opt.padd_size
        self.head = Conv2dBlock(3, nfc, (ker_size, ker_size), stride=stride, \
                                    padding=padd_size+num_layer, norm_fn="batchnorm", acti_fn="lrelu")

        body_layers = []
        for _ in range(num_layer-2):
            body_layers += [Conv2dBlock(
                nfc, nfc, (ker_size, ker_size), stride=stride, \
                                    padding=padd_size, norm_fn="batchnorm", acti_fn="lrelu"
            )]
        self.body = nn.SequentialCell(body_layers)

        self.tail = Conv2dBlock(nfc, 3, (ker_size, ker_size), stride=stride, \
                                    padding=padd_size, norm_fn="none", acti_fn="tanh")

    def construct(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + y

class Dis(nn.Cell):
    """Discriminator"""
    def __init__(self, opt):
        super().__init__()
        nfc = opt.nfc
        num_layer = opt.num_layer
        ker_size = opt.ker_size
        stride = opt.stride
        padd_size = opt.padd_size

        self.head = Conv2dBlock(3, nfc, (ker_size, ker_size), stride=stride, \
                                    padding=padd_size, norm_fn="none", acti_fn="lrelu")
        body_layers = []
        for _ in range(num_layer-2):
            body_layers += [Conv2dBlock(
                nfc, nfc, (ker_size, ker_size), stride=stride, \
                                    padding=padd_size, norm_fn="none", acti_fn="lrelu"
            )]
        self.body = nn.SequentialCell(body_layers)

        self.tail = Conv2dBlock(nfc, 1, (ker_size, ker_size), stride=stride, \
                                    padding=padd_size, norm_fn="none", acti_fn="none")

    def construct(self, x):
        """construct"""
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, (nn.GroupNorm, nn.BatchNorm2d)):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))

def get_model(model_idx, opt):
    """get netG and netD"""
    opt.nfc = opt.min_nfc * 2 ** (model_idx // 4)
    print('num_kernels=', opt.nfc)

    g = Gen(opt)
    d = Dis(opt)

    init_weights(g, 'KaimingUniform', math.sqrt(5))
    init_weights(d, 'KaimingUniform', math.sqrt(5))
    return g, d
