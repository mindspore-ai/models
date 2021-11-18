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
"""DeepID Model"""
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import set_seed

set_seed(1)

class DeepID(nn.Cell):
    """
    DeepID Module
    """
    def __init__(self, num_channels, class_num, feature=False):
        super(DeepID, self).__init__()
        self.feature = feature
        self.block1 = nn.SequentialCell(nn.Conv2d(in_channels=num_channels, out_channels=20,
                                                  kernel_size=4, has_bias=True, pad_mode='valid'),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=(2, 2)))
        self.block2 = nn.SequentialCell(nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3,
                                                  has_bias=True, pad_mode='valid'),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=(2, 2)))
        self.block3 = nn.SequentialCell(nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3,
                                                  has_bias=True, pad_mode='valid'),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=(2, 2)))
        self.deep_layer = nn.SequentialCell(nn.Conv2d(in_channels=60, out_channels=80, kernel_size=2,
                                                      has_bias=True, pad_mode='valid'),
                                            nn.ReLU())

        self.dense_layer = nn.SequentialCell(nn.Dense(in_channels=2160, out_channels=160),
                                             nn.ReLU())

        self.class_layer = nn.Dense(in_channels=160, out_channels=class_num)

        self.concat = P.Concat(axis=1)
        self.mode = 'train'


    def construct(self, x):
        "Forward"
        x = self.block1(x)

        x = self.block2(x)

        x1 = self.block3(x)
        x2 = self.deep_layer(x1)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)

        x = self.concat((x1, x2))
        out1 = self.dense_layer(x)
        if self.feature:
            return out1
        out = self.class_layer(out1)

        return out
