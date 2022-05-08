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

import math
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore.common.initializer as weight_init

groups_npu_1 = [[0, 1, 2, 3, 4, 5, 6, 7]]
groups_npu_4 = [[0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31]]
groups_npu_16 = [[0, 1, 2, 3, 4, 5, 6, 7],
                 [8, 9, 10, 11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20, 21, 22, 23],
                 [24, 25, 26, 27, 28, 29, 30, 31],
                 [32, 33, 34, 35, 36, 37, 38, 39],
                 [40, 41, 42, 43, 44, 45, 46, 47],
                 [48, 49, 50, 51, 52, 53, 54, 55],
                 [56, 57, 58, 59, 60, 61, 62, 63],
                 [64, 65, 66, 67, 68, 69, 70, 71],
                 [72, 73, 74, 75, 76, 77, 78, 79],
                 [80, 81, 82, 83, 84, 85, 86, 87],
                 [88, 89, 90, 91, 92, 93, 94, 95],
                 [96, 97, 98, 99, 100, 101, 102, 103],
                 [104, 105, 106, 107, 108, 109, 110, 111],
                 [112, 113, 114, 115, 116, 117, 118, 119],
                 [120, 121, 122, 123, 124, 125, 126, 127]]

def _make_divisible(x, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(x + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * x:
        new_v += divisor
    return new_v

class HardSwish(nn.Cell):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()
        self.mul = P.Mul()

    def construct(self, x):
        return self.mul(x, self.relu6(x + 3.)/6)

class MyHSigmoid(nn.Cell):
    def __init__(self):
        super(MyHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.) * 0.16666667


class Activation(nn.Cell):
    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = MyHSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise NotImplementedError

    def construct(self, x):
        return self.act(x)

class DropConnect(nn.Cell):
    def __init__(self, drop_connect_rate=0.):
        super(DropConnect, self).__init__()
        self.shape = P.Shape()
        self.dtype = P.DType()
        self.keep_prob = 1 - drop_connect_rate
        self.dropout = P.Dropout(keep_prob=self.keep_prob)

    def construct(self, x):
        shape = self.shape(x)
        dtype = self.dtype(x)
        ones_tensor = P.Fill()(dtype, (shape[0], 1, 1, 1), 1)
        mask, _ = self.dropout(ones_tensor)
        x = x * mask
        return x

def drop_connect(inputs, training=False, drop_connect_rate=0.):
    if not training or drop_connect_rate == 0:
        return inputs
    return DropConnect(drop_connect_rate)(inputs)

class GlobalAvgPooling(nn.Cell):
    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x

class SeModule(nn.Cell):
    def __init__(self, num_out, se_ratio=0.25, divisor=8):
        super(SeModule, self).__init__()
        num_mid = _make_divisible(num_out*se_ratio, divisor)
        self.pool = GlobalAvgPooling(keep_dims=True)
        self.conv_reduce = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                                     kernel_size=1, has_bias=True, pad_mode='pad')
        self.act1 = Activation('relu')
        self.conv_expand = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                                     kernel_size=1, has_bias=True, pad_mode='pad')
        self.act2 = Activation('hsigmoid')
        self.mul = P.Mul()

    def construct(self, x):
        out = self.pool(x)
        out = self.conv_reduce(out)
        out = self.act1(out)
        out = self.conv_expand(out)
        out = self.act2(out)
        out = self.mul(x, out)
        return out

class ConvBnAct(nn.Cell):
    def __init__(self, num_in, num_out, kernel_size, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu', sync_bn=False):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              group=num_groups,
                              has_bias=False,
                              pad_mode='pad')
        if sync_bn:
            self.bn = nn.SyncBatchNorm(num_out)
        else:
            self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else None

    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out

class GhostModule(nn.Cell):
    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu', sync_bn=False):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvBnAct(num_in, init_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                      num_groups=1, use_act=use_act, act_type=act_type, sync_bn=sync_bn)
        self.cheap_operation = ConvBnAct(init_channels, new_channels, kernel_size=dw_size, stride=1, padding=dw_size//2,
                                         num_groups=init_channels, use_act=use_act, act_type=act_type, sync_bn=sync_bn)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return self.concat((x1, x2))

class GhostBottleneck(nn.Cell):
    def __init__(self, num_in, num_mid, num_out, dw_kernel_size, stride=1,
                 act_type='relu', se_ratio=0., divisor=8, drop_path_rate=0., sync_bn=False):
        super(GhostBottleneck, self).__init__()

        use_se = se_ratio is not None and se_ratio > 0.
        self.drop_path_rate = drop_path_rate
        self.ghost1 = GhostModule(num_in, num_mid, kernel_size=1,
                                  stride=1, padding=0, act_type=act_type, sync_bn=sync_bn)

        self.use_dw = stride > 1
        if self.use_dw:
            self.dw = ConvBnAct(num_mid, num_mid, kernel_size=dw_kernel_size, stride=stride,
                                padding=(dw_kernel_size-1)//2, act_type=act_type,
                                num_groups=num_mid, use_act=False, sync_bn=sync_bn)

        self.use_se = use_se
        if use_se:
            self.se = SeModule(num_mid, se_ratio=se_ratio, divisor=divisor)

        self.ghost2 = GhostModule(num_mid, num_out, kernel_size=1, stride=1,
                                  padding=0, act_type=act_type, use_act=False, sync_bn=sync_bn)

        self.down_sample = False
        if num_in != num_out or stride != 1:
            self.down_sample = True

        if self.down_sample:
            self.shortcut = nn.SequentialCell([
                ConvBnAct(num_in, num_in, kernel_size=dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size-1)//2, num_groups=num_in, use_act=False, sync_bn=sync_bn),
                ConvBnAct(num_in, num_out, kernel_size=1, stride=1,
                          padding=0, num_groups=1, use_act=False, sync_bn=sync_bn),
            ])

        self.add = P.Add()

    def construct(self, x):
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.down_sample:
            shortcut = self.shortcut(shortcut)
        ## drop path
        if self.drop_path_rate > 0.:
            out = drop_connect(out, self.training, self.drop_path_rate)
        out = self.add(shortcut, out)
        return out

def gen_cfgs_1x(layers, channels):
    # generate configs of ghostnet
    cfgs = []
    for i in range(len(layers)):
        cfgs.append([])
        for j in range(layers[i]):
            if i == 0:
                cfgs[i].append([3, channels[i], channels[i], 0, 1])
            elif i == 1:
                if j == 0:
                    cfgs[i].append([3, channels[i-1]*3, channels[i], 0, 2])
                else:
                    cfgs[i].append([3, channels[i]*3, channels[i], 0, 1])
            elif i == 2:
                if j == 0:
                    cfgs[i].append([5, channels[i-1]*3, channels[i], 0.25, 2])
                else:
                    cfgs[i].append([5, channels[i]*3, channels[i], 0.25, 1])
            elif i == 3:
                if j == 0:
                    cfgs[i].append([3, channels[i-1]*6, channels[i], 0, 2])
                elif j == 1:
                    cfgs[i].append([3, 200, channels[i], 0, 1])
                else:
                    cfgs[i].append([3, 184, channels[i], 0, 1])
            elif i == 4:
                if j == 0:
                    cfgs[i].append([3, channels[i-1]*6, channels[i], 0.25, 1])
                else:
                    cfgs[i].append([3, channels[i]*6, channels[i], 0.25, 1])
            elif i == 5:
                if j == 0:
                    cfgs[i].append([5, channels[i-1]*6, channels[i], 0.25, 2])
                elif j%2 == 0:
                    cfgs[i].append([5, channels[i]*6, channels[i], 0.25, 1])
                else:
                    cfgs[i].append([5, channels[i]*6, channels[i], 0, 1])

    return cfgs


def gen_cfgs_large(layers, channels):
    # generate configs of ghostnet
    cfgs = []
    for i in range(len(layers)):
        cfgs.append([])
        if i == 0:
            for j in range(layers[i]):
                cfgs[i].append([3, channels[i], channels[i], 0.1, 1])
        elif i == 1:
            for j in range(layers[i]):
                if j == 0:
                    cfgs[i].append([3, channels[i-1]*3, channels[i], 0.1, 2])
                else:
                    cfgs[i].append([3, channels[i]*3, channels[i], 0.1, 1])
        elif i == 2:
            for j in range(layers[i]):
                if j == 0:
                    cfgs[i].append([5, channels[i-1]*3, channels[i], 0.1, 2])
                else:
                    cfgs[i].append([5, channels[i]*3, channels[i], 0.1, 1])
        elif i == 3:
            for j in range(layers[i]):
                if j == 0:
                    cfgs[i].append([3, channels[i-1]*6, channels[i], 0.1, 2])
                else:
                    cfgs[i].append([3, channels[i]*2.5, channels[i], 0.1, 1])
        elif i == 4:
            for j in range(layers[i]):
                if j == 0:
                    cfgs[i].append([3, channels[i-1]*6, channels[i], 0.1, 1])
                else:
                    cfgs[i].append([3, channels[i]*6, channels[i], 0.1, 1])
        elif i == 5:
            for j in range(layers[i]):
                if j == 0:
                    cfgs[i].append([5, channels[i-1]*6, channels[i], 0.1, 2])
                else:
                    cfgs[i].append([5, channels[i]*6, channels[i], 0.1, 1])
    return cfgs

class GhostNet(nn.Cell):
    def __init__(self, layers, channels, num_classes=1000, multiplier=1.,
                 final_drop=0., drop_path_rate=0., large=False, zero_init_residual=False, sync_bn=False):
        super(GhostNet, self).__init__()
        if layers is None:
            layers = [1, 2, 2, 4, 2, 5]
        if channels is None:
            channels = [16, 24, 40, 80, 112, 160]
        self.large = large
        if self.large:
            self.cfgs = gen_cfgs_large(layers, channels)
        else:
            self.cfgs = gen_cfgs_1x(layers, channels)

        self.drop_path_rate = drop_path_rate
        self.inplanes = 16
        first_conv_in_channel = 3
        first_conv_out_channel = _make_divisible(multiplier * channels[0], 4)

        self.conv_stem = nn.Conv2d(in_channels=first_conv_in_channel,
                                   out_channels=first_conv_out_channel,
                                   kernel_size=3, padding=1, stride=2,
                                   has_bias=False, pad_mode='pad')
        if sync_bn:
            self.bn1 = nn.SyncBatchNorm(first_conv_out_channel)
        else:
            self.bn1 = nn.BatchNorm2d(first_conv_out_channel)
        if self.large:
            self.act1 = HardSwish()
        else:
            self.act1 = Activation('relu')
        input_channel = first_conv_out_channel
        stages = []
        block = GhostBottleneck
        block_idx = 0
        block_count = sum(layers)
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * multiplier, 4)
                hidden_channel = _make_divisible(exp_size * multiplier, 4)
                drop_path_rate = self.drop_path_rate * block_idx / block_count
                if self.large:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s, act_type='relu',
                                        se_ratio=se_ratio, divisor=8, drop_path_rate=drop_path_rate, sync_bn=sync_bn))
                else:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s, act_type='relu',
                                        se_ratio=se_ratio, divisor=4, drop_path_rate=drop_path_rate, sync_bn=sync_bn))
                input_channel = output_channel
                block_idx += 1
                output_channel = _make_divisible(multiplier * exp_size, 4)
            stages.append(nn.SequentialCell(layers))

        if self.large:
            stages.append(ConvBnAct(input_channel, output_channel, 1, act_type='relu', sync_bn=sync_bn))  ###HardSuishMe
        else:
            stages.append(ConvBnAct(input_channel, output_channel, 1, act_type='relu', sync_bn=sync_bn))

        head_output_channel = max(1280, int(input_channel*5))
        input_channel = output_channel
        self.blocks = nn.SequentialCell(stages)

        self.global_pool = GlobalAvgPooling(keep_dims=True)
        self.conv_head = nn.Conv2d(input_channel,
                                   head_output_channel,
                                   kernel_size=1, padding=0, stride=1,
                                   has_bias=True, pad_mode='pad')
        if self.large:
            self.act2 = HardSwish()
        else:
            self.act2 = Activation('relu')
        self.squeeze = P.Flatten()
        self.final_drop = 1-final_drop
        if self.final_drop > 0:
            self.dropout = nn.Dropout(self.final_drop)

        self.classifier = nn.Dense(head_output_channel, num_classes, has_bias=True)

        self._initialize_weights()
        if zero_init_residual:
            for _, m in self.cellsand_names():
                if isinstance(m, GhostBottleneck):
                    tmp_x = Tensor(np.zeros(m.ghost2.primary_conv[1].weight.data.shape, dtype="float32"))
                    m.ghost2.primary_conv[1].weight.set_data(tmp_x)
                    tmp_y = Tensor(np.zeros(m.ghost2.cheap_operation[1].weight.data.shape, dtype="float32"))
                    m.ghost2.cheap_operation[1].weight.set_data(tmp_y)

    def construct(self, x):
        r"""construct of GhostNet"""
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.squeeze(x)
        if self.final_drop > 0:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                m.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                          m.weight.shape,
                                                          m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                          m.weight.shape,
                                                          m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
