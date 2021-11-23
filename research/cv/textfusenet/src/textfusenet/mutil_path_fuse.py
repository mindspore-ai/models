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
"""TextFuseNet mutil path fuse."""

import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore.ops import ReduceSum
from mindspore.ops import Stack


class MutilPathFuseModule(nn.Cell):
    """
        multi path fuse module for mask subnet

        Args:
            config(dict) - Config

        Returns:
            Tensor, output tensor
    """
    def __init__(self, cfg):
        super(MutilPathFuseModule, self).__init__()
        self.channels = cfg.textfusenet_channels
        self.num_expected_pos = cfg.num_expected_pos
        self.test_roi_number = cfg.test_roi_number
        self.mask_size = cfg.roi_layer.mask_out_size

        # char
        self.char_conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0,
                                      has_bias=False).to_float(mstype.float16)
        self.char_conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0,
                                      has_bias=False).to_float(mstype.float16)

        # text (word)
        self.text_conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0,
                                      has_bias=False).to_float(mstype.float16)
        self.text_conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0,
                                      has_bias=False).to_float(mstype.float16)

        # post processing
        self.conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0,
                                 has_bias=False).to_float(mstype.float16)
        self.conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0,
                                 has_bias=False).to_float(mstype.float16)

        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()

        # OP
        self.iou = P.IOU()
        self.reshape = P.Reshape()
        self.ones_value = Tensor(np.ones((cfg.num_expected_pos, cfg.num_expected_pos)).astype(np.float16))
        self.ones_value_test = Tensor(np.ones((cfg.test_roi_number, cfg.test_roi_number)).astype(np.float16))
        self.one_value = Tensor(1, mstype.int64)
        self.value = Tensor(1.0, mstype.float16)
        self.red_sum = ReduceSum()
        self.equal = P.Equal()
        self.tile = P.Tile()
        self.expand_dim = P.ExpandDims()
        self.logical_and = P.LogicalAnd()
        self.cast = P.Cast()
        self.greater = P.Greater()
        self.squeeze = P.Squeeze()
        self.stack = Stack()
        self.mul = P.Mul()
        self.trans = P.Transpose()
        self.softmax = P.Softmax(axis=1)
        self.argmax = P.ArgMaxWithValue(axis=1)

    def construct(self, x, classes, global_context, proposals):
        """mutil path fusion forward"""
        if self.training:
            roi_number = self.num_expected_pos
        else:
            roi_number = self.test_roi_number
        boxes = proposals[:, 1:]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area_box = w * h
        area_a = self.expand_dim(area_box, 0)
        area_b = self.expand_dim(area_box, 1)
        area_a = self.tile(area_a, (roi_number, 1))
        area_b = self.tile(area_b, (1, roi_number))
        iou_a_b = self.iou(boxes, boxes)
        area = area_a + area_b
        if self.training:
            inter_percent = iou_a_b*area/((self.ones_value+iou_a_b)*area_a)
        else:
            inter_percent = iou_a_b*area/((self.ones_value_test+iou_a_b)*area_a)
        # select the word(text)
        temp_index = self.equal(classes, self.one_value)
        temp_index = self.expand_dim(temp_index, 1)
        temp_index = self.cast(temp_index, mstype.float16)
        temp_index = self.tile(temp_index, (1, roi_number))
        char_pos = inter_percent > 0.9
        temp = char_pos
        # filter character
        temp_index = self.cast(temp_index, mstype.bool_)
        char_pos = self.logical_and(char_pos, temp_index)
        char_pos = self.cast(char_pos, mstype.float16)

        char_pos = self.expand_dim(char_pos, 2)
        char_pos = self.expand_dim(char_pos, 3)
        char_pos = self.tile(char_pos, (1, 1, self.channels, self.mask_size*self.mask_size))

        char_sum = self.red_sum(self.cast(temp, mstype.float16), 1)
        char_sum = self.expand_dim(char_sum, 1)
        char_sum = self.expand_dim(char_sum, 2)
        char_sum = self.expand_dim(char_sum, 3)

        char_sum = self.tile(char_sum, (1, self.channels, self.mask_size, self.mask_size))
        feat = self.reshape(x, (roi_number, self.channels, -1))
        feat = self.expand_dim(feat, 0)
        feat = self.tile(feat, (roi_number, 1, 1, 1))
        text_fuse_feature = feat * char_pos
        text_fuse_feature = self.red_sum(text_fuse_feature, 1)
        fuse_feature_shape = (roi_number,
                              self.channels,
                              self.mask_size,
                              self.mask_size)
        text_fuse_feature = self.reshape(text_fuse_feature, fuse_feature_shape)
        text_fuse_feature = text_fuse_feature / char_sum

        char_index = self.greater(classes, self.one_value)
        char_index = self.cast(char_index, mstype.float16)
        char_index = self.expand_dim(char_index, 1)
        char_index = self.expand_dim(char_index, 2)
        char_index = self.expand_dim(char_index, 3)
        char_index = self.tile(char_index, (1, self.channels, self.mask_size, self.mask_size))
        char_feature = char_index * x
        char_feature = self.char_conv3x3(char_feature)
        char_feature = self.char_conv1x1(char_feature)

        text_fuse_feature = self.text_conv3x3(text_fuse_feature)
        text_fuse_feature = self.text_conv1x1(text_fuse_feature)
        char_context = char_feature + text_fuse_feature

        feature_fuse = char_context + x
        feature_fuse = feature_fuse + global_context

        feature_fuse = self.conv3x3(feature_fuse)
        feature_fuse = self.conv1x1(feature_fuse)
        feature_fuse = self.bn(feature_fuse)
        feature_fuse = self.relu(feature_fuse)

        return feature_fuse
