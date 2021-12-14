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
"""Protonet"""
import mindspore.nn as nn
import mindspore.ops.functional


class protonet(nn.Cell):
    """Protonet"""
    def __init__(self):
        super(protonet, self).__init__()
        self.proto_conv1_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1)
        self.proto_conv2 = nn.Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
        self.proto_conv1_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1)
        self.proto_conv1_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1)
        self.proto_conv1_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1)
        self.relu = nn.ReLU()
        self.proto_bi = nn.ResizeBilinear()
        self.transpose = mindspore.ops.operations.Transpose()

    def construct(self, proto_x):
        """Forward"""
        proto_out = self.proto_conv1_1(proto_x)
        proto_out = self.relu(proto_out)
        proto_out = self.proto_conv1_2(proto_out)
        proto_out = self.relu(proto_out)
        proto_out = self.proto_conv1_3(proto_out)
        proto_out = self.relu(proto_out)
        proto_out = self.proto_bi(proto_out, scale_factor=2)
        proto_out = self.relu(proto_out)
        proto_out = self.proto_conv1_4(proto_out)
        proto_out = self.relu(proto_out)
        proto_out = self.proto_conv2(proto_out)
        perm = (0, 2, 3, 1)
        proto_out = self.transpose(proto_out, perm)
        return proto_out
