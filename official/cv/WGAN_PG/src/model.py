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

import mindspore.nn as nn
import mindspore.ops as ops

class DcgannobnD(nn.Cell):
    """ DCGAN Descriminator with no Batchnorm layer """
    def __init__(self, DIM):
        super(DcgannobnD, self).__init__()

        self.DIM = DIM
        KERNEL_SIZE = 5
        STRIDE = 2

        main = nn.SequentialCell()
        main.append(nn.Conv2d(3, self.DIM, KERNEL_SIZE, STRIDE, 'same'))
        main.append(nn.LeakyReLU(0.2))

        main.append(nn.Conv2d(self.DIM, self.DIM*2, KERNEL_SIZE, STRIDE, 'same'))
        main.append(nn.LeakyReLU(0.2))

        main.append(nn.Conv2d(self.DIM*2, self.DIM*4, KERNEL_SIZE, STRIDE, 'same'))
        main.append(nn.LeakyReLU(0.2))
        self.main = main
        self.linear = nn.Dense(4*4*4*self.DIM, 1)

    def construct(self, input1):

        output = self.main(input1)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        output = ops.reduce_mean(output)
        return output

class DcganG(nn.Cell):

    def __init__(self, DIM):
        super(DcganG, self).__init__()

        self.DIM = DIM
        KERNEL_SIZE = 5
        STRIDE = 2

        self.linear = nn.Dense(self.DIM, 4*4*4*self.DIM)
        self.bn = nn.BatchNorm2d(4*4*4*self.DIM)
        self.relu = nn.ReLU()

        main = nn.SequentialCell()
        main.append(nn.Conv2dTranspose(
            self.DIM*4,
            self.DIM*2,
            KERNEL_SIZE,
            stride=STRIDE,
            weight_init='normal',
            pad_mode='same'))
        main.append(nn.BatchNorm2d(self.DIM*2))
        main.append(nn.ReLU())

        main.append(nn.Conv2dTranspose(
            self.DIM*2,
            self.DIM,
            KERNEL_SIZE,
            stride=STRIDE,
            weight_init='normal',
            pad_mode='same'))
        main.append(nn.BatchNorm2d(self.DIM))
        main.append(nn.ReLU())

        main.append(nn.Conv2dTranspose(
            self.DIM,
            3,
            KERNEL_SIZE,
            stride=STRIDE,
            weight_init='normal',
            pad_mode='same'))
        main.append(nn.Tanh())
        self.main = main

    def construct(self, input1):

        output = self.linear(input1)
        output = output.view(64, 4*4*4*self.DIM, 1, 1)
        output = self.bn(output)
        output = self.relu(output)
        output = output.view(64, 4*self.DIM, 4, 4)
        output = self.main(output)
        return output
