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
"""
Loss function
"""

import mindspore.nn as nn


class ContentLoss(nn.Cell):
    """ContentLoss"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.nn_interpolate = nn.ResizeBilinear()

    def interpolate_downscale(self, x, scale_factor):
        """downscale"""
        _, _, h, w = x.shape
        h = h // scale_factor
        w = w // scale_factor
        return self.nn_interpolate(x, size=(h, w))

    def construct(self, pred_img, label_img):
        """construct ContentLoss"""
        label_img2 = self.interpolate_downscale(label_img, scale_factor=2)
        label_img4 = self.interpolate_downscale(label_img, scale_factor=4)
        l1 = self.criterion(pred_img[0], label_img4)
        l2 = self.criterion(pred_img[1], label_img2)
        l3 = self.criterion(pred_img[2], label_img)
        return l1+l2+l3
