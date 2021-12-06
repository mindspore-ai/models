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

"""pooler in vgg"""

import mindspore.nn as nn


class VGG(nn.Cell):
    """Use 4 pooler of VGG19 """

    def __init__(self):
        super(VGG, self).__init__()
        self.pool0 = nn.MaxPool2d(2, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

    def construct(self, x):
        """4 pooler compute graph
        Args:
            x(Tensor): low resolution image
        Outputs:
            Tensor
        """
        results = []
        x = self.pool0(x)
        results.append(x)
        x = self.pool1(x)
        results.append(x)
        x = self.pool2(x)
        results.append(x)
        x = self.pool3(x)
        results.append(x)
        return results


def vgg19():
    """return 4 pooler"""
    model = VGG()
    return model
