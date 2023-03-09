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
"""Model modules."""
import mindspore.numpy as msnp
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.ops import constexpr


class ScaledDotProductAttention(nn.Cell):
    """
    Scaled Dot-Product Attention.
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature

        self.softmax = nn.Softmax(axis=2)
        self.dropout = nn.Dropout(p=attn_dropout)

        self.bmm = ops.BatchMatMul()
        self.transpose = ops.Transpose()

    def construct(self, q, k, v, mask=None):
        """Forward."""
        attn = self.bmm(q, self.transpose(k, (0, 2, 1)))
        attn = attn / self.temperature

        inf_mask = infinity_mask(attn.shape, -msnp.inf)

        if mask is not None:
            attn = msnp.where(mask, inf_mask, attn)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = self.bmm(attn, v)

        return output


@constexpr
def infinity_mask(mask_shape, inf):
    """Make infinity mask."""
    inf_mask = ops.Fill()(mstype.float32, mask_shape, inf)
    return inf_mask
