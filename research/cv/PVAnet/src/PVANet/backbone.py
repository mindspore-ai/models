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

import os
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from mindspore.common.initializer import Normal, initializer

cat = ms.ops.Concat(1)


def _BatchNorm2dInit(out_chls, momentum=0.1, affine=True, use_batch_statistics=None):
    """Batchnorm2D wrapper."""
    dtype = np.float32
    gamma_init = ms.Tensor(np.array(np.ones(out_chls)).astype(dtype))
    beta_init = ms.Tensor(np.array(np.ones(out_chls) * 0).astype(dtype))
    moving_mean_init = ms.Tensor(np.array(np.ones(out_chls) * 0).astype(dtype))
    moving_var_init = ms.Tensor(np.array(np.ones(out_chls)).astype(dtype))
    return nn.BatchNorm2d(out_chls, momentum=momentum, affine=affine, gamma_init=gamma_init,
                          beta_init=beta_init, moving_mean_init=moving_mean_init,
                          moving_var_init=moving_var_init, use_batch_statistics=use_batch_statistics)


class _BnScaleRelu(nn.Cell):
    def __init__(self, channels):
        super(_BnScaleRelu, self).__init__()
        self.channels = channels
        self.bn = _BatchNorm2dInit(self.channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.bn(x)
        res = self.relu(x)
        return res


class _BnNegConcatScaleRelu(nn.Cell):
    def __init__(self, channels):
        super(_BnNegConcatScaleRelu, self).__init__()
        self.channels = channels
        self.bn_mean_var = _BatchNorm2dInit(self.channels, affine=False)
        self.bn_weight_bias = _BatchNorm2dInit(self.channels * 2, use_batch_statistics=False)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.bn_mean_var(x)
        concat_x = cat((x, -x))
        x = self.bn_weight_bias(concat_x)
        res = self.relu(x)
        return res


def weight_init_ones(shape):
    """Weight init."""
    # return ms.Tensor(np.full(shape, 0.01).astype(np.float32))

    n = shape[2] * shape[3] * shape[1]
    res = initializer(Normal(sigma=1., mean=math.sqrt(2. / n)), shape=shape).astype(np.float32)

    return res


class Conv1(nn.Cell):
    def __init__(self, in_channel=3, out_channel=16):
        super(Conv1, self).__init__()
        shape = (out_channel, in_channel, 7, 7)
        weights = weight_init_ones(shape)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               has_bias=False, weight_init=weights)
        self.bn_neg_cat_relu = _BnNegConcatScaleRelu(out_channel)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn_neg_cat_relu(x)
        x = self.pool(x)

        return x


class CReLUResidual(nn.Cell):
    def __init__(self, in_channels, conv1_channel, conv2_channel, conv3_channel, whether_2_3=2, if_first=False):
        super(CReLUResidual, self).__init__()
        self.if_first = if_first
        if whether_2_3 == 2:
            self.stride = 1
        elif whether_2_3 == 3:
            self.stride = 2
        if self.if_first:
            self.conv_proj = nn.Conv2d(in_channels, conv3_channel, kernel_size=1, stride=self.stride, pad_mode='pad',
                                       padding=0, has_bias=True,
                                       weight_init=weight_init_ones((conv3_channel, in_channels, 1, 1)))

        if not self.if_first:
            self.bn_scale_relu_1 = _BnScaleRelu(in_channels)
        self.bn_scale_relu_2 = _BnScaleRelu(conv1_channel)
        self.bn_neg_cat_relu = _BnNegConcatScaleRelu(conv2_channel)

        self.conv1 = nn.Conv2d(in_channels, conv1_channel, kernel_size=1, stride=self.stride, pad_mode='pad', padding=0,
                               has_bias=True, weight_init=weight_init_ones((conv1_channel, in_channels, 1, 1)))
        self.conv2 = nn.Conv2d(conv1_channel, conv2_channel, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                               has_bias=True, weight_init=weight_init_ones((conv2_channel, conv1_channel, 3, 3)))
        self.conv3 = nn.Conv2d(conv2_channel * 2, conv3_channel, kernel_size=1, stride=1, pad_mode='pad', padding=0,
                               has_bias=True, weight_init=weight_init_ones((conv3_channel, conv2_channel * 2, 1, 1)))

    def construct(self, x):
        original_x = x
        if not self.if_first:
            x = self.bn_scale_relu_1(x)

        x = self.conv1(x)
        x = self.bn_scale_relu_2(x)
        x = self.conv2(x)
        x = self.bn_neg_cat_relu(x)
        x = self.conv3(x)
        if self.if_first:
            original_x = self.conv_proj(original_x)
        res = x + original_x

        return res


class Inception(nn.Cell):
    def __init__(self, in_channels, conv1_1_param, conv3_3_params, conv5_5_params, out_channels, pool_param=None,
                 if_last=False):
        super(Inception, self).__init__()
        self.pool_param = pool_param
        self.out_channels = out_channels
        self.stride = 2 if pool_param else 1
        self.if_last = if_last

        self.bn_scale_relu = _BnScaleRelu(in_channels)

        self.branch1 = nn.SequentialCell(
            nn.Conv2d(in_channels, conv1_1_param, kernel_size=1, stride=self.stride, pad_mode='pad', padding=0,
                      has_bias=False, weight_init=weight_init_ones((conv1_1_param, in_channels, 1, 1))),
            _BnScaleRelu(conv1_1_param)
        )
        self.branch2 = nn.SequentialCell(
            nn.Conv2d(in_channels, conv3_3_params[0], kernel_size=1, stride=self.stride, pad_mode='pad', padding=0,
                      has_bias=False, weight_init=weight_init_ones((conv3_3_params[0], in_channels, 1, 1))),
            _BnScaleRelu(conv3_3_params[0]),
            nn.Conv2d(conv3_3_params[0], conv3_3_params[1], kernel_size=3, pad_mode='pad', padding=1, has_bias=False,
                      weight_init=weight_init_ones((conv3_3_params[1], conv3_3_params[0], 3, 3))),
            _BnScaleRelu(conv3_3_params[1])
        )
        self.branch3 = nn.SequentialCell(
            nn.Conv2d(in_channels, conv5_5_params[0], kernel_size=1, stride=self.stride, pad_mode='pad', padding=0,
                      has_bias=False, weight_init=weight_init_ones((conv5_5_params[0], in_channels, 1, 1))),
            _BnScaleRelu(conv5_5_params[0]),
            nn.Conv2d(conv5_5_params[0], conv5_5_params[1], kernel_size=3, pad_mode='pad', padding=1, has_bias=False,
                      weight_init=weight_init_ones((conv5_5_params[1], conv5_5_params[0], 3, 3))),
            _BnScaleRelu(conv5_5_params[1]),
            nn.Conv2d(conv5_5_params[1], conv5_5_params[2], kernel_size=3, pad_mode='pad', padding=1, has_bias=False,
                      weight_init=weight_init_ones((conv5_5_params[2], conv5_5_params[1], 3, 3))),
            _BnScaleRelu(conv5_5_params[2]),
        )
        self.concat_channels = conv1_1_param + conv3_3_params[1] + conv5_5_params[2]
        if pool_param:
            self.branch_pool = nn.SequentialCell(
                nn.MaxPool2d(kernel_size=3, stride=self.stride, pad_mode='same'),
                nn.Conv2d(in_channels, pool_param, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=False,
                          weight_init=weight_init_ones((pool_param, in_channels, 1, 1))),
                _BnScaleRelu(pool_param)
            )
            self.concat_channels += pool_param

        if self.pool_param:
            self.conv_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=2, padding=0,
                                       has_bias=True,
                                       weight_init=weight_init_ones((self.out_channels, in_channels, 1, 1)))
        if if_last:
            self.out_conv_last = nn.Conv2d(self.concat_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                           has_bias=False,
                                           weight_init=weight_init_ones(
                                               (self.out_channels, self.concat_channels, 1, 1)))
        else:
            self.out_conv = nn.Conv2d(self.concat_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                      has_bias=True,
                                      weight_init=weight_init_ones((self.out_channels, self.concat_channels, 1, 1)))

        if if_last:
            self.out_bn = _BatchNorm2dInit(out_channels)
            self.last_bn = _BnScaleRelu(out_channels)

    def construct(self, x):
        original_x = x
        x = self.bn_scale_relu(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        if self.pool_param:
            branch_pool = self.branch_pool(x)
            concat = cat((branch1, branch2, branch3, branch_pool))
        else:
            concat = cat((branch1, branch2, branch3))
        if self.if_last:
            x = self.out_conv_last(concat)
        else:
            x = self.out_conv(concat)

        if self.pool_param:
            original_x = self.conv_proj(original_x)
        if self.if_last:
            x = self.out_bn(x)
            x = x + original_x
            x = self.last_bn(x)
            return x

        res = x + original_x

        return res


class Pvanet(nn.Cell):
    def __init__(self):
        super(Pvanet, self).__init__()

        self.conv1 = Conv1()

        self.conv2 = nn.SequentialCell(
            CReLUResidual(32, 24, 24, 64, 2, if_first=True),
            CReLUResidual(64, 24, 24, 64),
            CReLUResidual(64, 24, 24, 64),
        )
        self.conv3 = nn.SequentialCell(
            CReLUResidual(64, 48, 48, 128, 3, if_first=True),
            CReLUResidual(128, 48, 48, 128),
            CReLUResidual(128, 48, 48, 128),
            CReLUResidual(128, 48, 48, 128),
        )
        self.conv4 = nn.SequentialCell(
            Inception(128, 64, [48, 128], [24, 48, 48], 256, 128),
            Inception(256, 64, [64, 128], [24, 48, 48], 256),
            Inception(256, 64, [64, 128], [24, 48, 48], 256),
            Inception(256, 64, [64, 128], [24, 48, 48], 256)
        )
        self.conv5 = nn.SequentialCell(
            Inception(256, 64, [96, 192], [32, 64, 64], 384, 128),
            Inception(384, 64, [96, 192], [32, 64, 64], 384),
            Inception(384, 64, [96, 192], [32, 64, 64], 384),
            Inception(384, 64, [96, 192], [32, 64, 64], 384, if_last=True),
        )

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # identity = ms.ops.stop_gradient(x)
        x1 = self.conv3(x)
        x2 = self.conv4(x1)
        x3 = self.conv5(x2)

        return x, x1, x2, x3


def get_pvanet(pretrained=True):
    from mindspore.train.serialization import load_checkpoint, load_param_into_net
    model = Pvanet()
    if pretrained:
        ckpt_path = './weights/pvanet_backbone.ckpt'
        if os.path.exists(ckpt_path):
            ms_model_param_dict = load_checkpoint(ckpt_path)
            load_param_into_net(model, ms_model_param_dict)
            print('Load pretrain success...')
        else:
            print('Load pretrain failed...')
    return model
