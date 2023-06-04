# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms


class Block(nn.Cell):
    def __init__(self, dim, dim_out, kernel_size, stride):
        super().__init__()
        self.act = nn.Conv2d(
            dim_out, dim_out, kernel_size=kernel_size,
            padding=kernel_size//2, pad_mode='pad', stride=stride, group=dim)
        self.conv = nn.Conv2d(dim, dim_out, 1)
        self.relu = nn.ReLU()
        self.pool = ms.ops.Identity() if stride == 1 else ms.ops.MaxPool(
            pad_mode="VALID", kernel_size=2, strides=stride)

    def construct(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.act(self.relu(x))
        return x


class VanillaNet(nn.Cell):
    def __init__(self, dims, block, strides, in_chans=3, num_classes=1000, kernel_size=7):
        super().__init__()
        self.depth = len(strides)

        self.stem = nn.SequentialCell(
            [
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                nn.ReLU(),
                nn.Conv2d(
                    dims[0], dims[0], kernel_size=kernel_size,
                    padding=kernel_size//2, pad_mode='pad', group=dims[0])
            ]
        )
        self.stages = nn.CellList()
        for i in range(self.depth):
            stage = block(
                dim=dims[i], dim_out=dims[i+1],
                kernel_size=kernel_size, stride=strides[i]).to_float(ms.float16)
            self.stages.append(stage)

        self.head = nn.Dense(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        for i in range(self.depth):
            x = self.stages[i](x)
        return x.mean([-2, -1])

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vanillanet_5():
    model = VanillaNet(block=Block, dims=[128*4, 256*4, 512*4, 1024*4], strides=[2, 2, 2])
    return model
