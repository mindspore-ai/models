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
I3D backbone
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal


class Unit3D(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0,
                 activation_fn=nn.ReLU, use_batch_norm=True, use_bias=False, name='unit_3d'):

        super(Unit3D, self).__init__()
        self.is_train = is_train
        self.amp_level = amp_level
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.op2 = ops.Concat(axis=2)
        self.op3 = ops.Concat(axis=3)
        self.op4 = ops.Concat(axis=4)
        self.zeros = ops.Zeros()
        if amp_level != 'O0':
            self.type = mindspore.float16
        else:
            self.type = mindspore.float32

        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                kernel_size=self._kernel_size, stride=self._stride, padding=0,
                                has_bias=self._use_bias, pad_mode='same', weight_init=HeNormal(mode='fan_out'))

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, gamma_init="ones", beta_init="zeros", momentum=0.1)
            if self.is_train:
                self.bn.set_train()

    def construct(self, x):

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            relu = nn.ReLU()
            x = relu(x)

        return x


class Unit3D_Conv3d_1a_7x7(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0,
                 activation_fn=nn.ReLU, use_batch_norm=True, use_bias=False, name='unit_3d'):

        super(Unit3D_Conv3d_1a_7x7, self).__init__()
        self.is_train = is_train
        self.amp_level = amp_level
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.op2 = ops.Concat(axis=2)
        self.op3 = ops.Concat(axis=3)
        self.op4 = ops.Concat(axis=4)
        self.zeros = ops.Zeros()
        if self.amp_level != 'O0':
            self.type = mindspore.float16
        else:
            self.type = mindspore.float32

        self.Conv3d_1a_7x7 = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                       kernel_size=self._kernel_size, stride=self._stride, padding=0,
                                       has_bias=self._use_bias, pad_mode='same', weight_init=HeNormal(mode='fan_out'))

        if self._use_batch_norm:
            self.Conv3d_1a_7x7_bn = nn.BatchNorm3d(self._output_channels, gamma_init="ones", beta_init="zeros",
                                                   momentum=0.1)
            if self.is_train:
                self.Conv3d_1a_7x7_bn.set_train()

    def construct(self, x):

        x = self.Conv3d_1a_7x7(x)
        if self._use_batch_norm:
            x = self.Conv3d_1a_7x7_bn(x)
        if self._activation_fn is not None:
            relu = nn.ReLU()
            x = relu(x)

        return x


class Unit3D_Conv3d_2b_1x1(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0,
                 activation_fn=nn.ReLU, use_batch_norm=True, use_bias=False, name='unit_3d'):

        super(Unit3D_Conv3d_2b_1x1, self).__init__()
        self.is_train = is_train
        self.amp_level = amp_level
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.op2 = ops.Concat(axis=2)
        self.op3 = ops.Concat(axis=3)
        self.op4 = ops.Concat(axis=4)
        self.zeros = ops.Zeros()
        if self.amp_level != 'O0':
            self.type = mindspore.float16
        else:
            self.type = mindspore.float32

        self.Conv3d_2b_1x1 = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                       kernel_size=self._kernel_size, stride=self._stride, padding=0,
                                       has_bias=self._use_bias, pad_mode='same', weight_init=HeNormal(mode='fan_out'))

        if self._use_batch_norm:
            self.Conv3d_2b_1x1_bn = nn.BatchNorm3d(self._output_channels, gamma_init="ones", beta_init="zeros",
                                                   momentum=0.1)
            if self.is_train:
                self.Conv3d_2b_1x1_bn.set_train()

    def construct(self, x):

        x = self.Conv3d_2b_1x1(x)
        if self._use_batch_norm:
            x = self.Conv3d_2b_1x1_bn(x)
        if self._activation_fn is not None:
            relu = nn.ReLU()
            x = relu(x)

        return x


class Unit3D_Conv3d_2c_3x3(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0,
                 activation_fn=nn.ReLU, use_batch_norm=True, use_bias=False, name='unit_3d'):

        super(Unit3D_Conv3d_2c_3x3, self).__init__()
        self.is_train = is_train
        self.amp_level = amp_level
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.op2 = ops.Concat(axis=2)
        self.op3 = ops.Concat(axis=3)
        self.op4 = ops.Concat(axis=4)
        self.zeros = ops.Zeros()
        if self.amp_level != 'O0':
            self.type = mindspore.float16
        else:
            self.type = mindspore.float32

        self.Conv3d_2c_3x3 = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                       kernel_size=self._kernel_size, stride=self._stride, padding=0,
                                       has_bias=self._use_bias, pad_mode='same', weight_init=HeNormal(mode='fan_out'))

        if self._use_batch_norm:
            self.Conv3d_2c_3x3_bn = nn.BatchNorm3d(self._output_channels, gamma_init="ones", beta_init="zeros",
                                                   momentum=0.1)
            if self.is_train:
                self.Conv3d_2c_3x3_bn.set_train()

    def construct(self, x):

        x = self.Conv3d_2c_3x3(x)
        if self._use_batch_norm:
            x = self.Conv3d_2c_3x3_bn(x)
        if self._activation_fn is not None:
            relu = nn.ReLU()
            x = relu(x)

        return x


class Unit3D_logits(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0,
                 activation_fn=nn.ReLU, use_batch_norm=False, use_bias=True, name='unit_3d'):

        super(Unit3D_logits, self).__init__()
        self.is_train = is_train
        self.amp_level = amp_level
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.op2 = ops.Concat(axis=2)
        self.op3 = ops.Concat(axis=3)
        self.op4 = ops.Concat(axis=4)
        self.zeros = ops.Zeros()
        if self.amp_level != 'O0':
            self.type = mindspore.float16
        else:
            self.type = mindspore.float32

        self.logits_conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels,
                                       kernel_size=self._kernel_size, stride=self._stride, padding=0,
                                       has_bias=self._use_bias, pad_mode='same', weight_init=HeNormal(mode='fan_out'))

        if self._use_batch_norm:
            self.logits_bn = nn.BatchNorm3d(self._output_channels, gamma_init="ones", beta_init="zeros", momentum=0.1)
            if self.is_train:
                self.logits_bn.set_train()

    def construct(self, x):

        x = self.logits_conv3d(x)
        if self._use_batch_norm:
            x = self.logits_bn(x)
        if self._activation_fn is not None:
            relu = nn.ReLU()
            x = relu(x)

        return x


class MaxPool3dSamePadding(nn.Cell):

    def __init__(self, amp_level, kernel_size, strieds):
        super(MaxPool3dSamePadding, self).__init__()
        self.maxpool = ops.MaxPool3D(kernel_size=kernel_size, strides=strieds, pad_mode='same')
        self.amp_level = amp_level
        self.strieds = strieds
        self.kernel_size = kernel_size
        self.op2 = ops.Concat(axis=2)
        self.op3 = ops.Concat(axis=3)
        self.op4 = ops.Concat(axis=4)
        self.zeros = ops.Zeros()
        if self.amp_level != 'O0':
            self.type = mindspore.float16
        else:
            self.type = mindspore.float32

    def construct(self, x):

        x = self.maxpool(x)

        return x


class UseMaxPool3D_replace_AvgPool3D(nn.Cell):

    def __init__(self, kernel_size, strieds):
        super(UseMaxPool3D_replace_AvgPool3D, self).__init__()
        self.uesmaxpool3d_replace_avgpool3d = ops.MaxPool3D(kernel_size=kernel_size, strides=strieds, pad_mode='valid')

    def construct(self, x):
        x = self.uesmaxpool3d_replace_avgpool3d(x)
        return x


class InceptionModule_Mixed_3b(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_3b, self).__init__()

        self.Mixed_3b_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_3b_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_3b_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_3b_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_3b_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_3b_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_3b_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_3b_b0(x)
        b1 = self.Mixed_3b_b1b(self.Mixed_3b_b1a(x))
        b2 = self.Mixed_3b_b2b(self.Mixed_3b_b2a(x))
        b3 = self.Mixed_3b_b3b(self.Mixed_3b_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_3c(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_3c, self).__init__()

        self.Mixed_3c_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_3c_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_3c_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_3c_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_3c_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_3c_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_3c_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_3c_b0(x)
        b1 = self.Mixed_3c_b1b(self.Mixed_3c_b1a(x))
        b2 = self.Mixed_3c_b2b(self.Mixed_3c_b2a(x))
        b3 = self.Mixed_3c_b3b(self.Mixed_3c_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_4b(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_4b, self).__init__()

        self.Mixed_4b_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_4b_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_4b_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_4b_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_4b_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_4b_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_4b_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_4b_b0(x)
        b1 = self.Mixed_4b_b1b(self.Mixed_4b_b1a(x))
        b2 = self.Mixed_4b_b2b(self.Mixed_4b_b2a(x))
        b3 = self.Mixed_4b_b3b(self.Mixed_4b_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_4c(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_4c, self).__init__()

        self.Mixed_4c_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_4c_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_4c_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_4c_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_4c_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_4c_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_4c_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_4c_b0(x)
        b1 = self.Mixed_4c_b1b(self.Mixed_4c_b1a(x))
        b2 = self.Mixed_4c_b2b(self.Mixed_4c_b2a(x))
        b3 = self.Mixed_4c_b3b(self.Mixed_4c_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_4d(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_4d, self).__init__()

        self.Mixed_4d_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_4d_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_4d_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_4d_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_4d_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_4d_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_4d_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_4d_b0(x)
        b1 = self.Mixed_4d_b1b(self.Mixed_4d_b1a(x))
        b2 = self.Mixed_4d_b2b(self.Mixed_4d_b2a(x))
        b3 = self.Mixed_4d_b3b(self.Mixed_4d_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_4e(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_4e, self).__init__()

        self.Mixed_4e_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_4e_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_4e_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_4e_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_4e_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_4e_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_4e_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_4e_b0(x)
        b1 = self.Mixed_4e_b1b(self.Mixed_4e_b1a(x))
        b2 = self.Mixed_4e_b2b(self.Mixed_4e_b2a(x))
        b3 = self.Mixed_4e_b3b(self.Mixed_4e_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_4f(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_4f, self).__init__()

        self.Mixed_4f_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_4f_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_4f_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_4f_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_4f_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_4f_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_4f_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_4f_b0(x)
        b1 = self.Mixed_4f_b1b(self.Mixed_4f_b1a(x))
        b2 = self.Mixed_4f_b2b(self.Mixed_4f_b2a(x))
        b3 = self.Mixed_4f_b3b(self.Mixed_4f_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_5b(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_5b, self).__init__()

        self.Mixed_5b_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_5b_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_5b_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_5b_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_5b_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_5b_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_5b_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_5b_b0(x)
        b1 = self.Mixed_5b_b1b(self.Mixed_5b_b1a(x))
        b2 = self.Mixed_5b_b2b(self.Mixed_5b_b2a(x))
        b3 = self.Mixed_5b_b3b(self.Mixed_5b_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionModule_Mixed_5c(nn.Cell):

    def __init__(self, is_train, amp_level, in_channels, out_channels, name):
        super(InceptionModule_Mixed_5c, self).__init__()

        self.Mixed_5c_b0 = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_0/Conv3d_0a_1x1')

        self.Mixed_5c_b1a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_1/Conv3d_0a_1x1')

        self.Mixed_5c_b1b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_1/Conv3d_0b_3x3')

        self.Mixed_5c_b2a = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_2/Conv3d_0a_1x1')

        self.Mixed_5c_b2b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + '/Branch_2/Conv3d_0b_3x3')

        self.Mixed_5c_b3a = MaxPool3dSamePadding(
            amp_level=amp_level,
            kernel_size=(3, 3, 3),
            strieds=(1, 1, 1))

        self.Mixed_5c_b3b = Unit3D(
            is_train=is_train,
            amp_level=amp_level,
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def construct(self, x):
        b0 = self.Mixed_5c_b0(x)
        b1 = self.Mixed_5c_b1b(self.Mixed_5c_b1a(x))
        b2 = self.Mixed_5c_b2b(self.Mixed_5c_b2a(x))
        b3 = self.Mixed_5c_b3b(self.Mixed_5c_b3a(x))

        concat_op = ops.Concat(axis=1)
        return concat_op([b0, b1, b2, b3])


class InceptionI3D(nn.Cell):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'logits',
    )

    def __init__(self, is_train=True, amp_level='O0', num_classes=400, train_spatial_squeeze=True,
                 final_endpoint='logits',
                 name='inception_i3d', in_channels=3, dropout_keep_prob=1.0, sample_duration=64):

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3D, self).__init__()

        self.is_train = is_train
        self.amp_level = amp_level
        self._model_name = name
        self._num_classes = num_classes
        self._train_spatial_squeeze = train_spatial_squeeze
        self._final_endpoint = final_endpoint
        self._dropout_keep_prob = dropout_keep_prob
        self._sample_duration = sample_duration
        self.squeeze_a = ops.Squeeze(3)
        self.squeeze_b = ops.Squeeze(2)

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.layers = {}
        end_point = 'Conv3d_1a_7x7'
        self.layers[end_point] = Unit3D_Conv3d_1a_7x7(self.is_train, self.amp_level, in_channels, 64,
                                                      kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=3,
                                                      name=name + end_point)

        end_point = 'MaxPool3d_2a_3x3'
        self.layers[end_point] = MaxPool3dSamePadding(self.amp_level, kernel_size=(1, 3, 3), strieds=(1, 2, 2))

        end_point = 'Conv3d_2b_1x1'
        self.layers[end_point] = Unit3D_Conv3d_2b_1x1(self.is_train, self.amp_level, 64, 64, kernel_size=(1, 1, 1),
                                                      padding=0, name=name + end_point)

        end_point = 'Conv3d_2c_3x3'
        self.layers[end_point] = Unit3D_Conv3d_2c_3x3(self.is_train, self.amp_level, 64, 192, kernel_size=(3, 3, 3),
                                                      padding=1, name=name + end_point)

        end_point = 'MaxPool3d_3a_3x3'
        self.layers[end_point] = MaxPool3dSamePadding(self.amp_level, kernel_size=(1, 3, 3), strieds=(1, 2, 2))

        end_point = 'Mixed_3b'
        self.layers[end_point] = InceptionModule_Mixed_3b(self.is_train, self.amp_level, 192, [64, 96, 128, 16, 32, 32],
                                                          name + end_point)

        end_point = 'Mixed_3c'
        self.layers[end_point] = InceptionModule_Mixed_3c(self.is_train, self.amp_level, 256,
                                                          [128, 128, 192, 32, 96, 64], name + end_point)

        end_point = 'MaxPool3d_4a_3x3'
        self.layers[end_point] = MaxPool3dSamePadding(self.amp_level, kernel_size=(3, 3, 3), strieds=(2, 2, 2))

        end_point = 'Mixed_4b'
        self.layers[end_point] = InceptionModule_Mixed_4b(self.is_train, self.amp_level, 128 + 192 + 96 + 64,
                                                          [192, 96, 208, 16, 48, 64], name + end_point)

        end_point = 'Mixed_4c'
        self.layers[end_point] = InceptionModule_Mixed_4c(self.is_train, self.amp_level, 192 + 208 + 48 + 64,
                                                          [160, 112, 224, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4d'
        self.layers[end_point] = InceptionModule_Mixed_4d(self.is_train, self.amp_level, 160 + 224 + 64 + 64,
                                                          [128, 128, 256, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4e'
        self.layers[end_point] = InceptionModule_Mixed_4e(self.is_train, self.amp_level, 128 + 256 + 64 + 64,
                                                          [112, 144, 288, 32, 64, 64], name + end_point)

        end_point = 'Mixed_4f'
        self.layers[end_point] = InceptionModule_Mixed_4f(self.is_train, self.amp_level, 112 + 288 + 64 + 64,
                                                          [256, 160, 320, 32, 128, 128], name + end_point)

        end_point = 'MaxPool3d_5a_2x2'
        self.layers[end_point] = MaxPool3dSamePadding(self.amp_level, kernel_size=(2, 2, 2), strieds=(2, 2, 2))

        end_point = 'Mixed_5b'
        self.layers[end_point] = InceptionModule_Mixed_5b(self.is_train, self.amp_level, 256 + 320 + 128 + 128,
                                                          [256, 160, 320, 32, 128, 128], name + end_point)

        end_point = 'Mixed_5c'
        self.layers[end_point] = InceptionModule_Mixed_5c(self.is_train, self.amp_level, 256 + 320 + 128 + 128,
                                                          [384, 192, 384, 48, 128, 128], name + end_point)

        end_point = 'AvgPool_5'
        self.layers[end_point] = UseMaxPool3D_replace_AvgPool3D((int(self._sample_duration / 8), 7, 7), (1, 1, 1))

        end_point = 'Dropout_5'
        self.layers[end_point] = nn.Dropout(self._dropout_keep_prob)

        end_point = 'logits'
        self.layers[end_point] = Unit3D_logits(self.is_train, self.amp_level, in_channels=384 + 384 + 128 + 128,
                                               output_channels=self._num_classes, kernel_size=(1, 1, 1), padding=0,
                                               activation_fn=None, use_batch_norm=False, use_bias=True,
                                               name=name + end_point)

        self._init_network()

    def _init_network(self):
        for layer_name, layer in self.layers.items():
            self.insert_child_to_cell(layer_name, layer)

    def construct(self, data):
        x = data
        x = self.layers['Conv3d_1a_7x7'](x)
        x = self.layers['MaxPool3d_2a_3x3'](x)
        x = self.layers['Conv3d_2b_1x1'](x)
        x = self.layers['Conv3d_2c_3x3'](x)
        x = self.layers['MaxPool3d_3a_3x3'](x)
        x = self.layers['Mixed_3b'](x)
        x = self.layers['Mixed_3c'](x)
        x = self.layers['MaxPool3d_4a_3x3'](x)
        x = self.layers['Mixed_4b'](x)
        x = self.layers['Mixed_4c'](x)
        x = self.layers['Mixed_4d'](x)
        x = self.layers['Mixed_4e'](x)
        x = self.layers['Mixed_4f'](x)
        x = self.layers['MaxPool3d_5a_2x2'](x)
        x = self.layers['Mixed_5b'](x)
        x = self.layers['Mixed_5c'](x)
        x = self.layers['AvgPool_5'](x)
        if self.is_train:
            x = self.layers['Dropout_5'](x)
        x = self.layers['logits'](x)

        output = self.squeeze_a(self.squeeze_a(x))
        if self._train_spatial_squeeze:
            output = self.squeeze_b(output)

        return output

    def trainable_params(self):
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.layers['logits'] = Unit3D_logits(self.is_train, self.amp_level, in_channels=384 + 384 + 128 + 128,
                                              output_channels=num_classes, kernel_size=(1, 1, 1), padding=0,
                                              activation_fn=None, use_batch_norm=False, use_bias=True,
                                              name=self._model_name + 'logits')

        self.logits = self.layers['logits']


def get_fine_tuning_parameters(model, ft_prefixes):
    assert isinstance(ft_prefixes, str)

    if ft_prefixes == '':
        return model.parameters()

    ft_prefixes = ft_prefixes.split(',')
    parameters = []
    param_names = []
    for param_name, param in model.parameters_and_names():
        for prefix in ft_prefixes:
            if param_name.startswith(prefix):
                parameters.append(param)
                param_names.append(param_name)

    for param_name, param in model.parameters_and_names():
        if param_name not in param_names:
            param.requires_grad = False

    return parameters
