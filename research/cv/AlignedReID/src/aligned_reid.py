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
""" AlignedReID model """

import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops

from src.resnet import resnet50


class AlignedReID(nn.Cell):
    """ AlignedReID network architecture. Predict global and local features of the image and classes probabilities

    Args:
        local_conv_out_channels: size of local feature vector
        num_classes: number of output classes (if 0 - remove last layer)
    """
    def __init__(self, local_conv_out_channels=128, num_classes=10):
        super().__init__()
        self.base = resnet50()
        planes = 2048
        self.num_classes = num_classes

        self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU()

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()

        if num_classes > 0:
            self.fc = nn.Dense(planes, num_classes)
        else:
            self.fc = mnp.zeros((1,))

    def construct(self, x):
        """ Forward """
        # shape [N, C, H, W]
        feat = self.base(x)

        global_feat = self.mean(feat, (2, 3))
        # shape [N, C]
        global_feat = self.flatten(global_feat)
        #         print('global_feat', global_feat.shape)

        # shape [N, C, H, 1]
        local_feat = self.mean(feat, 3)
        #         print('local_feat', local_feat.shape)

        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).transpose(0, 2, 1)
        #         print('local_feat2', local_feat.shape)

        if self.num_classes > 0:
            logits = self.fc(global_feat)
        else:
            logits = self.fc.copy()
        return global_feat, local_feat, logits
