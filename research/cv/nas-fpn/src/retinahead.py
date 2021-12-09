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

"""RetinaNet_Head."""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

class ClassificationModel(nn.Cell):
    """Classification head."""
    def __init__(self, num_features_in, num_anchors=9, num_classes=81, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, pad_mode='same')
        conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
        conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
        conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
        conv5 = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, pad_mode='same')

        self.feats = nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])
        self.transpose = P.Transpose()

    def construct(self, inputs):
        """Forward function."""
        batch_size = F.shape(inputs)[0]
        # out is B x C x W x H, with C = n_classes + n_anchors
        out = self.feats(inputs)
        # out1 is B x W x H x C
        out1 = self.transpose(out, (0, 2, 3, 1))
        batch_size, width, height, _ = out1.shape
        out2 = F.reshape(out1, (batch_size, width, height, self.num_anchors, self.num_classes))
        return F.reshape(out2, (batch_size, -1, self.num_classes))

class RegressionModel(nn.Cell):
    """Regression head."""
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, pad_mode='same')
        conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
        conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
        conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
        conv5 = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, pad_mode='same')

        self.feats = nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])
        self.transpose = P.Transpose()

    def construct(self, inputs):
        """Forward function."""
        batch_size = F.shape(inputs)[0]
        # out is B x C x W x H, with C = n_classes + n_anchors
        out = self.feats(inputs)
        # out1 is B x W x H x C
        out1 = self.transpose(out, (0, 2, 3, 1))
        return F.reshape(out1, (batch_size, -1, 4))

class RetinaNetHead(nn.Cell):
    """RetinaNet Head."""
    def __init__(self, config=None):
        super(RetinaNetHead, self).__init__()

        out_channels = config.extras_out_channels
        num_default = config.num_default[0]
        loc_layers = []
        cls_layers = []
        for out_channel in out_channels:
            loc_layers += [RegressionModel(num_features_in=out_channel, num_anchors=num_default)]
            cls_layers += [ClassificationModel(num_features_in=out_channel, num_anchors=num_default,
                                               num_classes=config.num_classes)]

        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.concat = P.Concat(axis=1)

    def construct(self, inputs):
        """Forward function."""
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.concat(loc_outputs), self.concat(cls_outputs)
