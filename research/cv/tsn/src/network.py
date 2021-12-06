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
"""changed bninception for tsn"""
import mindspore.nn as nn
import mindspore.ops as P

from mindspore.common.initializer import Normal

class First_layer(nn.Cell):
    """TSN first layer"""
    def __init__(self, modality):
        super(First_layer, self).__init__()

        self.modality = modality
        if self.modality == 'Flow':
            self.conv1_7x7_s2 = nn.Conv2d(10, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=True)
        elif self.modality == 'RGBDiff':
            self.conv1_7x7_s2 = nn.Conv2d(15, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=True)
        else:
            self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=True)

    def construct(self, x):
        """compute first layer"""
        return self.conv1_7x7_s2(x)

class Last_layer(nn.Cell):
    """TSN last layer"""
    def __init__(self, dropout, num_class):
        super(Last_layer, self).__init__()
        feature_dim = 1024
        self.num_class = num_class
        self.dropout = dropout
        self.reshape = P.Reshape()

        if self.dropout == 0:
            self.fc = nn.Dense(feature_dim, num_class)
            self.new_fc = None
        else:
            self.fc = nn.Dropout(keep_prob=self.dropout)
            self.new_fc = nn.Dense(feature_dim, num_class, weight_init=Normal(0.001))

    def construct(self, x):
        """compute last layer"""
        x = self.reshape(x, (x.shape[0], -1))
        x_out = self.fc(x)
        if self.new_fc:
            x_out = self.new_fc(x_out)
        return x_out


class BasicConv2d(nn.Cell):
    """TSN basicConv2d"""
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, pad_mode='pad', padding=0, has_bias=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, padding=padding, has_bias=has_bias)
        self.bn = nn.BatchNorm2d(out_channel, affine=False)
        self.relu = nn.ReLU()

    def construct(self, x):
        """compute basicConv2d"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception_3A(nn.Cell):
    """TSN Inception_3A"""
    def __init__(self):
        super(Inception_3A, self).__init__()
        self.inception_3a_1x1 = BasicConv2d(192, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_3a_3x3_reduce = BasicConv2d(192, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_3a_3x3 = BasicConv2d(64, 64, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3a_double_3x3_reduce = BasicConv2d(192, 64, kernel_size=1,\
             pad_mode='valid', stride=1, has_bias=True)
        self.inception_3a_double_3x3_1 = BasicConv2d(64, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3a_double_3x3_2 = BasicConv2d(96, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3a_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_3a_pool_proj = BasicConv2d(192, 32, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """compute Inception_3A"""
        inception_3a_relu_1x1_out = self.inception_3a_1x1(x)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_3x3_reduce(x)
        inception_3a_relu_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(x)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_pool_out = self.inception_3a_pool(x)
        inception_3a_relu_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_output_out = self.concate((inception_3a_relu_1x1_out, inception_3a_relu_3x3_out,\
             inception_3a_relu_double_3x3_2_out, inception_3a_relu_pool_proj_out))
        return inception_3a_output_out


class Inception_3B(nn.Cell):
    """TSN Inception_3B"""
    def __init__(self,):
        super(Inception_3B, self).__init__()
        self.inception_3b_1x1 = BasicConv2d(256, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_3b_3x3_reduce = BasicConv2d(256, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_3b_3x3 = BasicConv2d(64, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3b_double_3x3_reduce = BasicConv2d(256, 64, kernel_size=1,\
             stride=1, pad_mode='valid', has_bias=True)
        self.inception_3b_double_3x3_1 = BasicConv2d(64, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3b_double_3x3_2 = BasicConv2d(96, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3b_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_3b_pool_proj = BasicConv2d(256, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_3B"""
        inception_3b_relu_1x1_out = self.inception_3b_1x1(x)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_3x3_reduce(x)
        inception_3b_relu_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(x)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_pool_out = self.inception_3b_pool(x)
        inception_3b_relu_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_output_out = self.concate((inception_3b_relu_1x1_out, inception_3b_relu_3x3_out,\
             inception_3b_relu_double_3x3_2_out, inception_3b_relu_pool_proj_out))
        return inception_3b_output_out


class Inception_3C(nn.Cell):
    """TSN Inception_3C"""
    def __init__(self):
        super(Inception_3C, self).__init__()
        self.inception_3c_3x3_reduce = BasicConv2d(320, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_3c_3x3 = BasicConv2d(128, 160, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True)
        self.inception_3c_double_3x3_reduce = BasicConv2d(320, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_3c_double_3x3_1 = BasicConv2d(64, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3c_double_3x3_2 = BasicConv2d(96, 96, kernel_size=3, stride=2,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_3c_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_3C"""
        inception_3c_relu_3x3_reduce_out = self.inception_3c_3x3_reduce(x)
        inception_3c_relu_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(x)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_pool_out = self.inception_3c_pool(x)
        inception_3c_output_out = self.concate((inception_3c_relu_3x3_out,\
             inception_3c_relu_double_3x3_2_out, inception_3c_pool_out))
        return inception_3c_output_out


class Inception_4A(nn.Cell):
    """TSN Inception_4A"""
    def __init__(self):
        super(Inception_4A, self).__init__()
        self.inception_4a_1x1 = BasicConv2d(576, 224, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4a_3x3_reduce = BasicConv2d(576, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4a_3x3 = BasicConv2d(64, 96, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4a_double_3x3_reduce = BasicConv2d(576, 96, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4a_double_3x3_1 = BasicConv2d(96, 128, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4a_double_3x3_2 = BasicConv2d(128, 128, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4a_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_4a_pool_proj = BasicConv2d(576, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_4A"""
        inception_4a_relu_1x1_out = self.inception_4a_1x1(x)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_3x3_reduce(x)
        inception_4a_relu_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(x)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_pool_out = self.inception_4a_pool(x)
        inception_4a_relu_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_output_out = self.concate((inception_4a_relu_1x1_out, inception_4a_relu_3x3_out,\
             inception_4a_relu_double_3x3_2_out, inception_4a_relu_pool_proj_out))
        return inception_4a_output_out


class Inception_4B(nn.Cell):
    """TSN Inception_4B"""
    def __init__(self):
        super(Inception_4B, self).__init__()
        self.inception_4b_1x1 = BasicConv2d(576, 192, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4b_3x3_reduce = BasicConv2d(576, 96, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4b_3x3 = BasicConv2d(96, 128, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4b_double_3x3_reduce = BasicConv2d(576, 96, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4b_double_3x3_1 = BasicConv2d(96, 128, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4b_double_3x3_2 = BasicConv2d(128, 128, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4b_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_4b_pool_proj = BasicConv2d(576, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_4B"""
        inception_4b_relu_1x1_out = self.inception_4b_1x1(x)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_3x3_reduce(x)
        inception_4b_relu_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(x)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_pool_out = self.inception_4b_pool(x)
        inception_4b_relu_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_output_out = self.concate((inception_4b_relu_1x1_out, inception_4b_relu_3x3_out,\
             inception_4b_relu_double_3x3_2_out, inception_4b_relu_pool_proj_out))
        return inception_4b_output_out


class Inception_4C(nn.Cell):
    """TSN Inception_4C"""
    def __init__(self):
        super(Inception_4C, self).__init__()
        self.inception_4c_1x1 = BasicConv2d(576, 160, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4c_3x3_reduce = BasicConv2d(576, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4c_3x3 = BasicConv2d(128, 160, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4c_double_3x3_reduce = BasicConv2d(576, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4c_double_3x3_1 = BasicConv2d(128, 160, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4c_double_3x3_2 = BasicConv2d(160, 160, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4c_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_4c_pool_proj = BasicConv2d(576, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_4C"""
        inception_4c_relu_1x1_out = self.inception_4c_1x1(x)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_3x3_reduce(x)
        inception_4c_relu_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(x)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_pool_out = self.inception_4c_pool(x)
        inception_4c_relu_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_output_out = self.concate((inception_4c_relu_1x1_out, inception_4c_relu_3x3_out,\
             inception_4c_relu_double_3x3_2_out, inception_4c_relu_pool_proj_out))
        return inception_4c_output_out


class Inception_4D(nn.Cell):
    """TSN Inception_4D"""
    def __init__(self):
        super(Inception_4D, self).__init__()
        self.inception_4d_1x1 = BasicConv2d(608, 96, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4d_3x3_reduce = BasicConv2d(608, 128, kernel_size=1, stride=1, pad_mode='valid', has_bias=True)
        self.inception_4d_3x3 = BasicConv2d(128, 192, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4d_double_3x3_reduce = BasicConv2d(608, 160, kernel_size=1,\
             stride=1, pad_mode='valid', has_bias=True)
        self.inception_4d_double_3x3_1 = BasicConv2d(160, 192, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4d_double_3x3_2 = BasicConv2d(192, 192, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4d_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_4d_pool_proj = BasicConv2d(608, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_4D"""
        inception_4d_relu_1x1_out = self.inception_4d_1x1(x)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_3x3_reduce(x)
        inception_4d_relu_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(x)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_pool_out = self.inception_4d_pool(x)
        inception_4d_relu_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_output_out = self.concate((inception_4d_relu_1x1_out, inception_4d_relu_3x3_out,\
             inception_4d_relu_double_3x3_2_out, inception_4d_relu_pool_proj_out))
        return inception_4d_output_out


class Inception_4E(nn.Cell):
    """TSN Inception_4E"""
    def __init__(self):
        super(Inception_4E, self).__init__()
        self.inception_4e_3x3_reduce = BasicConv2d(608, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4e_3x3 = BasicConv2d(128, 192, kernel_size=3, stride=2,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4e_double_3x3_reduce = BasicConv2d(608, 192, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_4e_double_3x3_1 = BasicConv2d(192, 256, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4e_double_3x3_2 = BasicConv2d(256, 256, kernel_size=3, stride=2,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_4e_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_4E"""
        inception_4e_relu_3x3_reduce_out = self.inception_4e_3x3_reduce(x)
        inception_4e_relu_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(x)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_pool_out = self.inception_4e_pool(x)
        inception_4e_output_out = self.concate((inception_4e_relu_3x3_out,\
             inception_4e_relu_double_3x3_2_out, inception_4e_pool_out))
        return inception_4e_output_out


class Inception_5A(nn.Cell):
    """TSN Inception_5A"""
    def __init__(self):
        super(Inception_5A, self).__init__()
        self.inception_5a_1x1 = BasicConv2d(1056, 352, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_5a_3x3_reduce = BasicConv2d(1056, 192, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_5a_3x3 = BasicConv2d(192, 320, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_5a_double_3x3_reduce = BasicConv2d(1056, 160, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_5a_double_3x3_1 = BasicConv2d(160, 224, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_5a_double_3x3_2 = BasicConv2d(224, 224, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_5a_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_5a_pool_proj = BasicConv2d(1056, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_5A"""
        inception_5a_relu_1x1_out = self.inception_5a_1x1(x)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_3x3_reduce(x)
        inception_5a_relu_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(x)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_pool_out = self.inception_5a_pool(x)
        inception_5a_relu_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_output_out = self.concate((inception_5a_relu_1x1_out, inception_5a_relu_3x3_out,\
             inception_5a_relu_double_3x3_2_out, inception_5a_relu_pool_proj_out))
        return inception_5a_output_out


class Inception_5B(nn.Cell):
    """TSN Inception_5A"""
    def __init__(self):
        super(Inception_5B, self).__init__()
        self.inception_5b_1x1 = BasicConv2d(1024, 352, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_5b_3x3_reduce = BasicConv2d(1024, 192, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_5b_3x3 = BasicConv2d(192, 320, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_5b_double_3x3_reduce = BasicConv2d(1024, 192, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.inception_5b_double_3x3_1 = BasicConv2d(192, 224, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_5b_double_3x3_2 = BasicConv2d(224, 224, kernel_size=3, stride=1,\
             pad_mode='pad', padding=1, has_bias=True)
        self.inception_5b_pool = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.inception_5b_pool_proj = BasicConv2d(1024, 128, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """Compute Inception_5A"""
        inception_5b_relu_1x1_out = self.inception_5b_1x1(x)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_3x3_reduce(x)
        inception_5b_relu_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(x)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_pool_out = self.inception_5b_pool(x)
        inception_5b_relu_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_output_out = self.concate((inception_5b_relu_1x1_out, inception_5b_relu_3x3_out,\
             inception_5b_relu_double_3x3_2_out, inception_5b_relu_pool_proj_out))
        return inception_5b_output_out


class BNInception(nn.Cell):
    """BNInception"""
    def __init__(self, num_class, modality, dropout):
        super(BNInception, self).__init__()

        self.conv1_7x7_s2 = First_layer(modality)
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU()

        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.conv2_3x3_reduce = BasicConv2d(64, 64, kernel_size=1, stride=1,\
             pad_mode='valid', has_bias=True)
        self.conv2_3x3 = BasicConv2d(64, 192, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)

        self.pool2_3x3_s2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.inception_3a = Inception_3A()
        self.inception_3b = Inception_3B()
        self.inception_3c = Inception_3C()
        self.inception_4a = Inception_4A()
        self.inception_4b = Inception_4B()
        self.inception_4c = Inception_4C()
        self.inception_4d = Inception_4D()
        self.inception_4e = Inception_4E()
        self.inception_5a = Inception_5A()
        self.inception_5b = Inception_5B()

        self.fc = Last_layer(num_class=num_class, dropout=dropout)
        self.global_pool = nn.AvgPool2d(kernel_size=7)

    def construct(self, x):
        """BNInception"""
        conv1_7x7_s2_out = self.conv1_7x7_s2(x)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_relu_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_relu_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_output_out = self.inception_3a(pool2_3x3_s2_out)
        inception_3b_output_out = self.inception_3b(inception_3a_output_out)
        inception_3c_output_out = self.inception_3c(inception_3b_output_out)
        inception_4a_output_out = self.inception_4a(inception_3c_output_out)
        inception_4b_output_out = self.inception_4b(inception_4a_output_out)
        inception_4c_output_out = self.inception_4c(inception_4b_output_out)
        inception_4d_output_out = self.inception_4d(inception_4c_output_out)
        inception_4e_output_out = self.inception_4e(inception_4d_output_out)
        inception_5a_output_out = self.inception_5a(inception_4e_output_out)
        inception_5b_output_out = self.inception_5b(inception_5a_output_out)

        x = self.global_pool(inception_5b_output_out)
        x = self.fc(x)
        return x
