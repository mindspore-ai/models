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
"""Tokenizer Function"""
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

from src.models.cct.misc import Identity


class Tokenizer(nn.Cell):
    """Tokenizer"""
    def __init__(self, kernel_size, stride, padding, pooling_kernel_size=3, pooling_stride=2, n_conv_layers=1,
                 n_input_channels=3, n_output_channels=64, in_planes=64, activation=None, max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        conv_layers = []
        for i in range(n_conv_layers):
            conv_layers.append(nn.SequentialCell(
                [nn.Conv2d(n_filter_list[i], n_filter_list[i + 1], kernel_size=(kernel_size, kernel_size),
                           stride=(stride, stride), padding=padding, has_bias=conv_bias, pad_mode="pad"),
                 Identity() if activation is None else activation(),
                 nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, pad_mode="same") if max_pool else
                 Identity()]
            ))
        self.conv_layers = nn.CellList(conv_layers)

        self.flattener = ops.Reshape()

    def sequence_length(self, n_channels=3, height=224, width=224):
        """get sequence length"""
        return self.construct(Tensor(np.zeros((1, n_channels, height, width)), mstype.float32)).shape[1]

    def construct(self, x):
        for cell in self.conv_layers:
            x = cell(x)
        x = self.flattener(x, (x.shape[0], x.shape[1], -1))
        x = ops.Transpose()(x, (0, 2, 1))
        return x
