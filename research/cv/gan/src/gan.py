# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
'''train the gan model'''
from src.loss import GenWithLossCell
from src.loss import DisWithLossCell
import numpy as np
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore.common import initializer
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C

class Reshape(nn.Cell):
    def __init__(self, shape, auto_prefix=True):
        super().__init__(auto_prefix=auto_prefix)
        self.shape = shape
        self.reshape = P.Reshape()

    def construct(self, x):
        return self.reshape(x, self.shape)

class InstanceNorm2d(nn.Cell):
    """InstanceNorm2d"""

    def __init__(self, channel):
        super(InstanceNorm2d, self).__init__()
        self.gamma = Parameter(initializer.initializer(
            init=Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32)), shape=[1, channel, 1, 1]),
                               name='gamma')
        self.beta = Parameter(initializer.initializer(init=initializer.Zero(), shape=[1, channel, 1, 1]),
                              name='beta')
        self.reduceMean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sub = P.Sub()
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.tile = P.Tile()
        self.reshape = P.Reshape()
        self.eps = Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32) * 1e-5)
        self.cast2fp32 = P.Cast()

    def construct(self, x):
        mean = self.reduceMean(x, (2, 3))
        mean_stop_grad = F.stop_gradient(mean)
        variance = self.reduceMean(self.square(self.sub(x, mean_stop_grad)), (2, 3))
        variance = variance + self.eps
        inv = self.rsqrt(variance)
        normalized = self.sub(x, mean) * inv
        x_IN = self.add(self.mul(self.gamma, normalized), self.beta)
        return x_IN

class Generator(nn.Cell):
    """generator"""

    def __init__(self, latent_size, auto_prefix=True):
        super(Generator, self).__init__(auto_prefix=auto_prefix)
        self.network = nn.SequentialCell()

        self.network.append(nn.Dense(latent_size, 256 * 7 * 7, has_bias=False))
        self.network.append(Reshape((-1, 256, 7, 7)))
        self.network.append(InstanceNorm2d(256))
        self.network.append(nn.ReLU())

        self.network.append(nn.Conv2dTranspose(256, 128, 5, 1))
        self.network.append(InstanceNorm2d(128))
        self.network.append(nn.ReLU())

        self.network.append(nn.Conv2dTranspose(128, 64, 5, 2))
        self.network.append(InstanceNorm2d(64))
        self.network.append(nn.ReLU())

        self.network.append(nn.Conv2dTranspose(64, 1, 5, 2))
        self.network.append(nn.Tanh())

    def construct(self, x):
        return self.network(x)


class Discriminator(nn.Cell):
    '''discriminator'''

    def __init__(self, auto_prefix=True):
        super().__init__(auto_prefix=auto_prefix)
        self.network = nn.SequentialCell()

        self.network.append(nn.Conv2d(1, 64, 5, 2))
        self.network.append(InstanceNorm2d(64))
        self.network.append(nn.LeakyReLU())

        self.network.append(nn.Conv2d(64, 128, 5, 2))
        self.network.append(InstanceNorm2d(128))
        self.network.append(nn.LeakyReLU())

        self.network.append(nn.Flatten())
        self.network.append(nn.Dense(128 * 7 * 7, 1))

    def construct(self, x):
        return self.network(x)


class TrainOneStepCell(nn.Cell):
    '''TrainOneStepCell'''

    def __init__(
            self,
            netG: GenWithLossCell,
            netD: DisWithLossCell,
            optimizerG1: nn.Optimizer,
            optimizerD1: nn.Optimizer,
            sens=1.0,
            auto_prefix=True,
    ):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netG.set_grad()
        self.netG.add_flags(defer_inline=True)

        self.netD = netD
        self.netD.set_grad()
        self.netD.add_flags(defer_inline=True)

        self.weights_G = optimizerG1.parameters
        self.optimizerG1 = optimizerG1
        self.weights_D = optimizerD1.parameters
        self.optimizerD1 = optimizerD1

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens

        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity

    # train discriminator
    def trainD(self, real_data, latent_code3, loss, loss_net, grad, optimizer,
               weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = grad(loss_net, weights)(real_data, latent_code3, sens)
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss

    # train generator
    def trainG(self, latent_code4, loss, loss_net, grad, optimizer, weights,
               grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = grad(loss_net, weights)(latent_code4, sens)
        grads = grad_reducer(grads)
        optimizer(grads)
        return loss

    def construct(self, real_data, latent_code5):
        '''construct'''
        loss_D = self.netD(real_data, latent_code5)
        loss_G = self.netG(latent_code5)
        d_out2, g_out2 = None, None
        d_out2 = self.trainD(real_data, latent_code5, loss_D, self.netD, self.grad, self.optimizerD1,
                             self.weights_D, self.grad_reducer_D)
        g_out2 = self.trainG(latent_code5, loss_G, self.netG, self.grad, self.optimizerG1,
                             self.weights_G, self.grad_reducer_G)
        return d_out2, g_out2
