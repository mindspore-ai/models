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

"""Customized eval net"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P


class CTCEvalModel(nn.Cell):
    """
          customized eval net
           Args:
               network(Cell): trained network
    """

    def __init__(self, network):
        super(CTCEvalModel, self).__init__()
        self.network = network
        self.cast = ops.Cast()
        self.shapes = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, feature, masks, label, seq_len):
        logits = self.network(feature, masks)
        return logits, label, seq_len
