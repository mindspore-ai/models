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
from mindspore import nn
import mindspore
from src.utils import mul, reshape, eye


class ResBlock(nn.Cell):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, has_bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.SequentialCell(*m)
        self.res_scale = res_scale

    def construct(self, x):
        res = mul(self.body(x), self.res_scale)
        res += x

        return res


class BasicBlock(nn.SequentialCell):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU()):
        m = [conv(in_channels, out_channels, kernel_size, has_bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, stride=1, has_bias=True):
    c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size // 2), has_bias=has_bias,
                  pad_mode='pad')
    return c


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = mindspore.Tensor(rgb_std)
        self.weight = mindspore.Parameter((reshape(eye(3), (3, 3, 1, 1)) / reshape(std, (3, 1, 1, 1))), name='w')
        # self.weight = self.weight.astype(mindspore.dtype.float16)
        self.bias = mindspore.Parameter(sign * rgb_range * mindspore.Tensor(rgb_mean) / std, name='b')
        for p in self.get_parameters():
            p.requires_grad = False
