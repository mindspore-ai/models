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
""" Part-based Convolutional Baseline with refined part pooling network """

from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import HeNormal, Normal, Constant

from src.model_utils.config import config
from src.resnet import resnet50


class RPP(nn.Cell):
    """ Part-based Convolutional Baseline with refined part pooling network

    Args:
        num_classes: number of output classes
    """
    def __init__(self, num_classes):
        super().__init__()

        self.part = 6
        self.num_classes = num_classes

        resnet = resnet50(1000)
        self.base = resnet

        self.dropout = nn.Dropout(keep_prob=0.5)

        # self.avg_pool3d = ops.AvgPool3D(kernel_size=(8,1,1),strides=(8,1,1),pad_mode="valid")
        self.avg_pool_2 = ops.AvgPool(kernel_size=(8, 1), strides=(8, 1), pad_mode="valid")
        self.local_mask = nn.Conv2d(256, 6, kernel_size=1, pad_mode="valid", has_bias=True,
                                    weight_init=HeNormal(mode="fan_out"), bias_init=Constant(0))

        self.local_conv = nn.Conv2d(2048, 256, kernel_size=1, has_bias=False, pad_mode="valid",
                                    weight_init=HeNormal(mode="fan_out"))
        self.feat_bn2d = nn.BatchNorm2d(num_features=256)

        self.relu = ops.ReLU()
        self.expand_dims = ops.ExpandDims()
        self.trans = ops.Transpose()
        self.norm = nn.Norm(axis=1)

        # self.squeeze = ops.Squeeze(1)
        self.avgpool2d = ops.AvgPool(kernel_size=(24, 8))
        self.softmax = ops.Softmax(axis=1)
        self.split_1 = ops.Split(1, 6)
        self.split_2 = ops.Split(2, 6)
        self.concat = ops.Concat(2)
        # define 6 classifiers
        self.instance0 = nn.Dense(256, self.num_classes, weight_init=Normal(sigma=0.001), bias_init=Constant(0))
        self.instance1 = nn.Dense(256, self.num_classes, weight_init=Normal(sigma=0.001), bias_init=Constant(0))
        self.instance2 = nn.Dense(256, self.num_classes, weight_init=Normal(sigma=0.001), bias_init=Constant(0))
        self.instance3 = nn.Dense(256, self.num_classes, weight_init=Normal(sigma=0.001), bias_init=Constant(0))
        self.instance4 = nn.Dense(256, self.num_classes, weight_init=Normal(sigma=0.001), bias_init=Constant(0))
        self.instance5 = nn.Dense(256, self.num_classes, weight_init=Normal(sigma=0.001), bias_init=Constant(0))

    def construct(self, x):
        """ Forward """
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.pad(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        y = x
        if self.training:
            y = self.dropout(x)

        y = self.trans(y, (0, 2, 1, 3))
        y = self.avg_pool_2(y)
        y = self.trans(y, (0, 2, 1, 3))

        center = self.avgpool2d(y)
        y = y - center.expand_as(y)
        local_mask = self.local_mask(y)
        local_mask = self.softmax(local_mask)

        lw = self.split_1(local_mask)
        x = x * 6
        f0 = x * (lw[0].expand_as(x))
        f1 = x * (lw[1].expand_as(x))
        f2 = x * (lw[2].expand_as(x))
        f3 = x * (lw[3].expand_as(x))
        f4 = x * (lw[4].expand_as(x))
        f5 = x * (lw[5].expand_as(x))
        f0 = self.avgpool2d(f0)
        f1 = self.avgpool2d(f1)
        f2 = self.avgpool2d(f2)
        f3 = self.avgpool2d(f3)
        f4 = self.avgpool2d(f4)
        f5 = self.avgpool2d(f5)
        x = self.concat((f0, f1, f2, f3, f4, f5))

        feat = self.concat((f0, f1, f2, f3, f4, f5))
        g_feature = feat / (self.expand_dims(self.norm(feat), 1)).expand_as(feat)

        if self.training:
            x = self.dropout(x)
        x = self.local_conv(x)

        h_feature = x / (self.expand_dims(self.norm(x), 1)).expand_as(x)
        x = self.feat_bn2d(x)

        x = self.relu(x)

        x = self.split_2(x)
        x0 = x[0].view(x[0].shape[0], -1)
        x1 = x[1].view(x[1].shape[0], -1)
        x2 = x[2].view(x[2].shape[0], -1)
        x3 = x[3].view(x[3].shape[0], -1)
        x4 = x[4].view(x[4].shape[0], -1)
        x5 = x[5].view(x[5].shape[0], -1)

        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        c4 = self.instance4(x4)
        c5 = self.instance5(x5)

        return (c0, c1, c2, c3, c4, c5), g_feature, h_feature


use_g_feature = config.use_G_feature


class RPPInfer(nn.Cell):
    """ Part-based Convolutional Baseline with refined part pooling inference network
    """
    def __init__(self):
        super().__init__()

        self.part = 6
        resnet = resnet50(1000)
        self.base = resnet

        # self.avg_pool3d = ops.AvgPool3D(kernel_size=(8,1,1),strides=(8,1,1),pad_mode="valid")
        self.avg_pool_2 = ops.AvgPool(kernel_size=(8, 1), strides=(8, 1), pad_mode="valid")
        self.local_mask = nn.Conv2d(256, 6, kernel_size=1, pad_mode="valid", has_bias=True,
                                    weight_init=HeNormal(mode="fan_out"), bias_init=Constant(0))

        self.local_conv = nn.Conv2d(2048, 256, kernel_size=1, has_bias=False, pad_mode="valid",
                                    weight_init=HeNormal(mode="fan_out"))
        self.feat_bn2d = nn.BatchNorm2d(num_features=256)

        self.relu = ops.ReLU()
        self.expand_dims = ops.ExpandDims()
        self.trans = ops.Transpose()
        self.norm = nn.Norm(axis=1)

        self.avgpool2d = ops.AvgPool(kernel_size=(24, 8))
        self.softmax = ops.Softmax(axis=1)
        self.split_1 = ops.Split(1, 6)
        self.concat = ops.Concat(2)

    def construct(self, x):
        """ Forward """
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.pad(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        y = x

        y = self.trans(y, (0, 2, 1, 3))
        y = self.avg_pool_2(y)
        y = self.trans(y, (0, 2, 1, 3))

        center = self.avgpool2d(y)
        y = y - center.expand_as(y)
        local_mask = self.local_mask(y)
        local_mask = self.softmax(local_mask)

        lw = self.split_1(local_mask)
        x = x * 6
        f0 = x * (lw[0].expand_as(x))
        f1 = x * (lw[1].expand_as(x))
        f2 = x * (lw[2].expand_as(x))
        f3 = x * (lw[3].expand_as(x))
        f4 = x * (lw[4].expand_as(x))
        f5 = x * (lw[5].expand_as(x))
        f0 = self.avgpool2d(f0)
        f1 = self.avgpool2d(f1)
        f2 = self.avgpool2d(f2)
        f3 = self.avgpool2d(f3)
        f4 = self.avgpool2d(f4)
        f5 = self.avgpool2d(f5)
        x = self.concat((f0, f1, f2, f3, f4, f5))

        feat = self.concat((f0, f1, f2, f3, f4, f5))
        g_feature = feat / (self.expand_dims(self.norm(feat), 1)).expand_as(feat)

        x = self.local_conv(x)

        h_feature = x / (self.expand_dims(self.norm(x), 1)).expand_as(x)

        if use_g_feature:
            return g_feature
        return h_feature
