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
# ===========================================================================

"""
    Initialize network weights.
"""

import functools
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore.common import initializer as init
from src.utils.config import config
from .discriminator_model import MultiscaleDiscriminator
from .generator_model import GlobalGenerator, LocalEnhancer, Encoder


class InstanceNorm2d(nn.Cell):
    """InstanceNorm2d"""

    def __init__(self, channel):
        super(InstanceNorm2d, self).__init__()
        self.gamma = ms.Parameter(
            init.initializer(
                init=ms.Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32)), shape=[1, channel, 1, 1]
            ),
            name="gamma",
            requires_grad=False,
        )

        self.beta = ms.Parameter(
            init.initializer(init=init.Zero(), shape=[1, channel, 1, 1]), name="beta", requires_grad=False
        )
        self.reduceMean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.sub = ops.Sub()
        self.add = ops.Add()
        self.rsqrt = ops.Rsqrt()
        self.mul = ops.Mul()
        self.eps = ms.Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32) * 1e-5)

    def construct(self, x):
        mean = self.reduceMean(x, (2, 3))
        mean_stop_grad = ops.stop_gradient(mean)
        variance = self.reduceMean(self.square(self.sub(x, mean_stop_grad)), (2, 3))
        variance = variance + self.eps
        inv = self.rsqrt(variance)
        normalized = self.sub(x, mean) * inv
        x_IN = self.add(self.mul(self.gamma, normalized), self.beta)
        return x_IN


def init_weights(net, init_type="normal", init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == "normal":
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == "xavier":
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == "constant":
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer("ones", cell.gamma.shape))
            cell.beta.set_data(init.initializer("zeros", cell.beta.shape))


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = InstanceNorm2d
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_generator(
    input_nc,
    output_nc,
    ngf,
    netG,
    n_downsample_global=3,
    n_blocks_global=9,
    n_local_enhancers=1,
    n_blocks_local=3,
    norm="instance",
):
    """
    Return a generator by args.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "global":
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == "local":
        netG = LocalEnhancer(
            input_nc,
            output_nc,
            ngf,
            n_downsample_global,
            n_blocks_global,
            n_local_enhancers,
            n_blocks_local,
            norm_layer,
        )
    elif netG == "encoder":
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise NotImplementedError("generator [%s] is not found" % netG)

    init_weights(netG, init_type=config.init_type, init_gain=config.init_gain)
    return netG


def get_discriminator(input_nc, ndf, n_layer_D, norm="instance", use_sigmoid=False, num_D=1, getIntermFeat=False):
    """
    Return a discriminator by args.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layer_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    init_weights(netD, init_type=config.init_type, init_gain=config.init_gain)
    return netD
