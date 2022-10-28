# Copyright 2020 Huawei Technologies Co., Ltd
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
""" define some operator """
import mindspore
from mindspore import Tensor
import mindspore.ops as P
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import constexpr
import numpy as np


class MaskedFill(nn.Cell):
    """ masked fill """

    def __init__(self):
        super().__init__()
        self.select = P.Select()
        self.fill = P.Fill()
        self.cast = P.Cast()

    def construct(self, inputs: Tensor, mask: Tensor, value):
        """

        Args:
            inputs:
            mask:
            value:

        Returns:

        """
        mask = self.cast(mask, mstype.bool_)
        masked_value = self.fill(mstype.float32, inputs.shape, value)
        output = self.select(mask, masked_value, inputs)
        return output


class Matmul(nn.Cell):
    """ matmul """

    def __init__(self):
        super(Matmul, self).__init__()
        self.cast = P.Cast()
        self.matmul = nn.MatMul()

    def construct(self, input1, input2):
        """

        Args:
            input1:
            input2:

        Returns:

        """
        input1_16 = self.cast(input1, mstype.float16)
        input2_16 = self.cast(input2, mstype.float16)
        output = self.matmul(input1_16, input2_16)
        output = self.cast(output, mstype.float32)
        return output


@constexpr
def generate_arange_tensor(seq_length):
    np_array = np.arange(seq_length)
    return Tensor(np_array, mindspore.int32)


@constexpr
def generate_arange_tensor_start(seq_length, padding_idx):
    np_array = np.arange(padding_idx + 1, seq_length + padding_idx + 1)
    return Tensor(np_array, mindspore.int32)
