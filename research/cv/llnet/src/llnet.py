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
"""LLNet model definition"""
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import Parameter

layer_sizes = [867, 578, 289]

#Sparse Denoising Autoencoder
class SDA(nn.Cell):
    def __init__(self, input_shape=289, output_shape=867, w1=None, b1=None, b1_=None,
                 pretrain=False, pretrain_corrupted_level=0.1):
        super(SDA, self).__init__()
        if w1 is None:
            self.w1 = Parameter(Tensor(np.random.uniform(-4.0 * np.sqrt(6.0 / (input_shape + output_shape)),
                                                         4.0 * np.sqrt(6.0 / (input_shape + output_shape)),
                                                         [input_shape, output_shape]),
                                       mindspore.float32))
        else:
            self.w1 = Parameter(Tensor(w1), mindspore.float32)
        if b1 is None:
            self.b1 = Parameter(Tensor(np.zeros(output_shape), mindspore.float32))
        else:
            self.b1 = Parameter(Tensor(b1), mindspore.float32)
        if b1_ is None:
            self.b1_ = Parameter(Tensor(np.zeros(input_shape), mindspore.float32))
        else:
            self.b1_ = Parameter(Tensor(b1_), mindspore.float32)
        self.transpose = ops.Transpose()
        self.w1_ = self.transpose(self.w1, (1, 0))
        self.pretrain = pretrain
        self.pretrain_corrupted_level = pretrain_corrupted_level
        self.activation = ops.Sigmoid()
        self.matmul = ops.MatMul()
        self.zero = Tensor(0.0, mindspore.float32)
        self.stddev = Tensor(self.pretrain_corrupted_level, mindspore.float32)
        self.float16 = False

    def construct(self, inputs):
        if self.pretrain:
            if self.pretrain_corrupted_level > 0.0:
                noise = ops.normal(shape=(1, inputs.shape[-1]), mean=self.zero, stddev=self.stddev)
                x = noise + inputs
                x = x.clip(0, 1)
            else:
                x = inputs
            y1 = self.activation(self.matmul(x, self.w1) + self.b1)
            z1 = self.activation(self.matmul(y1, self.w1_) + self.b1_)
            return z1

        if self.float16:
            x_16 = inputs.astype(mstype.float16)
            w1_16 = self.w1.astype(mstype.float16)
            b1_16 = self.b1.astype(mstype.float16)
            y1 = self.activation(self.matmul(x_16, w1_16) + b1_16)
            return y1.astype(mstype.float32)

        x = inputs
        y1 = self.activation(self.matmul(x, self.w1) + self.b1)
        return y1

class SDA_WithLossCell(nn.Cell):
    def __init__(self, backbone):
        super(SDA_WithLossCell, self).__init__()
        self.backbone = backbone
        self.loss_function = nn.MSELoss()

    def construct(self, inputs, targets):
        z1 = self.backbone(inputs)
        loss_value = self.loss_function(z1, targets)
        return loss_value

class SDA_TrainOneStepCell(nn.Cell):
    """customized trainning"""

    def __init__(self, network, optimizer):
        """
            Args:
                 network: the network which will be trained.
                 optimizer: for updating parameters.
        """
        super(SDA_TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                                  # the forward network
        self.network.set_grad()                                 # the backward network
        self.optimizer = optimizer                              # the optimizer
        self.weights = self.optimizer.parameters                # the weights
        self.grad = ops.GradOperation(get_by_list=True)         # the gradient function

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # calculate the loss
        grads = self.grad(self.network, self.weights)(*inputs)  # update the gradient
        self.optimizer(grads)                                   # update the weights
        return loss

#Stacked Sparse Denoising Autoencoder with 6 layers
class LLNet(nn.Cell):
    def __init__(self, input_shape=289):
        super(LLNet, self).__init__()
        self.da1 = SDA(input_shape=input_shape, output_shape=layer_sizes[0])
        self.da2 = SDA(input_shape=layer_sizes[0], output_shape=layer_sizes[1])
        self.da3 = SDA(input_shape=layer_sizes[1], output_shape=layer_sizes[2])
        self.da4 = SDA(input_shape=layer_sizes[2], output_shape=layer_sizes[1])
        self.da5 = SDA(input_shape=layer_sizes[1], output_shape=layer_sizes[0])
        self.da6 = SDA(input_shape=layer_sizes[0], output_shape=289)

    def initial_decoder(self, w1s=None, b1s=None, b1_s=None):
        if w1s is not None:
            w1 = w1s[0]
            b1 = b1s[0]
            b1_ = b1_s[0]
            self.da4.w1 = Parameter(w1, 'da4.w1')
            self.da4.b1 = Parameter(b1, 'da4.b1')
            self.da4.b1_ = Parameter(b1_, 'da4.b1_')

            w1 = w1s[1]
            b1 = b1s[1]
            b1_ = b1_s[1]
            self.da5.w1 = Parameter(w1, 'da5.w1')
            self.da5.b1 = Parameter(b1, 'da5.b1')
            self.da5.b1_ = Parameter(b1_, 'da5.b1_')

            w1 = w1s[2]
            b1 = b1s[2]
            b1_ = b1_s[2]
            self.da6.w1 = Parameter(w1, 'da6.w1')
            self.da6.b1 = Parameter(b1, 'da6.b1')
            self.da6.b1_ = Parameter(b1_, 'da6.b1_')

    def set_float16(self):
        self.da1.float16 = True
        self.da2.float16 = True
        self.da3.float16 = True
        self.da4.float16 = True
        self.da5.float16 = True
        self.da6.float16 = True

    def calculate_outputs(self, inputs):
        x = self.da1(inputs)
        x = self.da2(x)
        x = self.da3(x)
        x = self.da4(x)
        x = self.da5(x)
        x = self.da6(x)
        return x

    def construct(self, inputs):
        return self.calculate_outputs(inputs)
