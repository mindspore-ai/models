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
# ===========================================================================
"""mindx_utils"""
import sys
import os

import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops

from src.utils.utils import BuildEvalNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class InferWithFlipNetwork(nn.Cell):
    """InferWithFlipNetwork"""
    def __init__(self, network, flip=True, input_format="NCHW"):
        super(InferWithFlipNetwork, self).__init__()
        self.eval_net = BuildEvalNetwork(network)
        self.transpose = ops.Transpose()
        self.flip = flip
        self.format = input_format

    def construct(self, input_data):
        """construct"""
        if self.format == "NHWC":
            input_data = self.transpose(input_data, (0, 3, 1, 2))
        output = self.eval_net(input_data)

        if self.flip:
            flip_input = msnp.flip(input_data, 3)
            flip_output = self.eval_net(flip_input)
            output += msnp.flip(flip_output, 3)

        return output
