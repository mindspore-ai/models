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

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor


class CoarseNet(nn.Cell):
    def __init__(self, init_weights=True):
        super(CoarseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, pad_mode="same", has_bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding=(1, 1, 2, 2), pad_mode="pad", has_bias=True
        )
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode="same", has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, pad_mode="same", has_bias=True)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, pad_mode="same", has_bias=True)
        self.deconv6 = nn.Conv2dTranspose(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, pad_mode="pad", has_bias=True
        )
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, pad_mode="same", has_bias=True)
        self.deconv8 = nn.Conv2dTranspose(
            in_channels=128, out_channels=64, kernel_size=2, stride=2, pad_mode="pad", has_bias=True
        )
        self.conv9 = nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=3, padding=(1, 0, 0, 0), pad_mode="pad", has_bias=True
        )
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, pad_mode="same", has_bias=True)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, pad_mode="same", has_bias=True)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, pad_mode="same", has_bias=True)
        self.fc13 = nn.Dense(in_channels=4070, out_channels=4070)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = P.Concat(axis=1)
        self.flatten = nn.Flatten()
        self.reshape = P.Reshape()
        self.dropout = nn.Dropout(p=0.5)

    def construct(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.pool(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.pool(x3)

        x4 = self.conv4(x3)
        x4 = self.relu(x4)
        x4 = self.pool(x4)

        x5 = self.conv5(x4)
        x5 = self.relu(x5)

        x5 = self.conv5(x5)
        x5 = self.relu(x5)

        x5 = self.conv5(x5)
        x5 = self.relu(x5)

        x6 = self.deconv6(x5)
        x6 = self.relu(x6)

        x7 = self.concat((x3, x6))
        x7 = self.conv7(x7)
        x7 = self.relu(x7)

        x8 = self.deconv8(x7)
        x8 = self.relu(x8)
        x8 = self.concat((x2, x8))

        x9 = self.conv9(x8)
        x9 = self.relu(x9)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x12 = self.flatten(x12)
        x13 = self.fc13(x12)
        x14 = self.reshape(x13, (-1, 1, 55, 74))
        return x14


class FineNet(nn.Cell):
    def __init__(self, init_weights=True):
        super(FineNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=63, kernel_size=9, stride=2, padding=0, pad_mode="pad")
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, pad_mode="pad")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2, pad_mode="pad")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        if init_weights:
            self._initialize_weights()

    def construct(self, x, y):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.concat((x, y))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.001, m.weight.data.shape).astype("float32")))
                m.bias.set_data(Tensor(np.random.normal(0, 0.001, m.bias.data.shape).astype("float32")))
