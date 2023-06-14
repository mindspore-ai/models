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
"""OneStepCell"""
import mindspore.nn as nn


class OneStepCell(nn.Cell):
    """OneStepCell"""

    def __init__(self, network):
        super(OneStepCell, self).__init__()
        self.network = network

        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="valid")

    def construct(self, img):
        output = self.network(img)

        output_one = self.pool(self.pad(output[0]))
        output_two = self.pool(self.pad(output[1]))

        return [output_one, output_two]
