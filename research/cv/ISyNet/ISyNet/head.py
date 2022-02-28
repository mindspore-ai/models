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
"""ISyNet head"""
from mindspore import nn
from mindspore import ops

__all__ = ['CustomHead']

class CustomHead(nn.Cell):
    """IsyNet classification head"""
    custom_layers = ()
    def __init__(self, num_classes, channels_out, dropout, last_bn):
        super().__init__()
        self.last_bn = last_bn
        out_channels = 1280

        self._conv_head = nn.Conv2d(channels_out, out_channels, kernel_size=1, has_bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.9, eps=1e-5)
        self._relu1 = nn.ReLU()
        # Final linear layer
        self._avg_pooling = ops.ReduceMean(keep_dims=False)
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self._dropout = nn.Dropout(dropout)
        self._fc = nn.Dense(out_channels, num_classes)
        if self.last_bn:
            self._bn2 = nn.BatchNorm2d(num_features=num_classes, momentum=0.9, eps=1e-5)
        self.view = ops.Reshape()

    def construct(self, *inputs, **_kwargs):
        """ISyNet head construct"""
        x = inputs[0]
        bs = x.shape[0]

        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._avg_pooling(x, (2, 3))
        x = self.view(x, (bs, -1))
        if self.use_dropout:
            x = self._dropout(x)
        x = self._fc(x)
        if self.last_bn:
            # nn.BatchNorm1d is broken so we reshape data to (N, C, 1, 1) to use nn.BatchNorm2d
            x = self.view(x, (bs, -1, 1, 1))
            x = self._bn2(x)
            x = self.view(x, (bs, -1))

        return x
