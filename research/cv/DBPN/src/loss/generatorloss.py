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

"""GENERATOR_LOSS"""

import mindspore.nn as nn


class GeneratorLoss(nn.Cell):
    """Loss for srresnet
    use GeneratorLoss to measure the L1Loss
    Args:
        generator: use the generator to rebuild sr image
    Outputs:
        Tensor
    """

    def __init__(self, generator):
        super(GeneratorLoss, self).__init__()
        self.generator = generator
        self.criterion = nn.L1Loss()

    def construct(self, HR_img, LR_img):
        """compute mse loss between sr and hr """
        hr = HR_img
        sr = self.generator(LR_img)
        content_loss = self.criterion(hr, sr)
        return content_loss
