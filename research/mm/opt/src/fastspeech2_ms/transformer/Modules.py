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
"""Fastspeech2 Modules"""
from mindspore import nn, ops


class ScaledDotProductAttention(nn.Cell):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(axis=2)
        self.bmm = ops.BatchMatMul()

    def construct(self, q, k, v, mask=None):
        """ScaledDotProductAttention construct"""
        attn = self.bmm(q, k.transpose(0, 2, 1))

        attn = attn / self.temperature

        attn = self.softmax(attn)
        output = self.bmm(attn, v)

        return output, attn
