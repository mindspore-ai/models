# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""rcan"""
import math
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common import Tensor, Parameter


def default_conv(in_channels, out_channels, kernel_size, has_bias=True):
    """rcan"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), has_bias=has_bias, pad_mode='pad')


class MeanShift(nn.Conv2d):
    """rcan"""
    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):
        """rcan"""
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.reshape = P.Reshape()
        self.eye = P.Eye()
        std = Tensor(rgb_std, mstype.float32)
        self.weight.set_data(
            self.reshape(self.eye(3, 3, mstype.float32), (3, 3, 1, 1)) / self.reshape(std, (3, 1, 1, 1)))
        self.weight.requires_grad = False
        self.bias = Parameter(
            sign * rgb_range * Tensor(rgb_mean, mstype.float32) / std, name='bias', requires_grad=False)
        self.has_bias = True



class PixelShuffle(nn.Cell):
    """PixelShuffle"""
    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def construct(self, x):
        n, c, ih, iw = x.shape
        oh = ih * self.scale
        ow = iw * self.scale
        oc = c // (self.scale ** 2)
        output = P.Transpose()(x, (0, 2, 1, 3))
        output = P.Reshape()(output, (n, ih, oc * self.scale, self.scale, iw))
        output = P.Transpose()(output, (0, 1, 2, 4, 3))
        output = P.Reshape()(output, (n, ih, oc, self.scale, ow))
        output = P.Transpose()(output, (0, 2, 1, 3, 4))
        output = P.Reshape()(output, (n, oc, oh, ow))
        return output



class Upsampler(nn.Cell):
    """rcan"""
    def __init__(self, conv, scale, n_feats, has_bias=True):
        """rcan"""
        super(Upsampler, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, has_bias))
                m.append(PixelShuffle(2))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, has_bias))
        else:
            raise NotImplementedError

        self.net = nn.SequentialCell(m)

    def construct(self, x):
        """rcan"""
        return self.net(x)

class AdaptiveAvgPool2d(nn.Cell):
    """rcan"""
    def __init__(self):
        """rcan"""
        super().__init__()
        self.ReduceMean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        """rcan"""
        return self.ReduceMean(x, (2, 3))

class CALayer(nn.Cell):
    """rcan"""
    def __init__(self, channel, reduction=16):
        """rcan"""
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = AdaptiveAvgPool2d()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.SequentialCell([
            nn.Conv2d(channel, channel // reduction, 1, padding=0, has_bias=True, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, has_bias=True, pad_mode='pad'),
            nn.Sigmoid()
        ])

    def construct(self, x):
        """rcan"""
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Cell):
    """rcan"""
    def __init__(self, conv, n_feat, kernel_size, reduction, has_bias=True
                 , bn=False, act=nn.ReLU(), res_scale=1):
        """rcan"""
        super(RCAB, self).__init__()
        self.modules_body = []
        for i in range(2):
            self.modules_body.append(conv(n_feat, n_feat, kernel_size, has_bias=has_bias))
            if bn: self.modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: self.modules_body.append(act)
        self.modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.SequentialCell(*self.modules_body)
        self.res_scale = res_scale

    def construct(self, x):
        """rcan"""
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Cell):
    """rcan"""
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        """rcan"""
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, has_bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.SequentialCell(*modules_body)

    def construct(self, x):
        """rcan"""
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Cell):
    """rcan"""
    def __init__(self, args, conv=default_conv):
        """rcan"""
        super(RCAN, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        self.dytpe = mstype.float16

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks)\
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.SequentialCell(modules_head)
        self.body = nn.SequentialCell(modules_body)
        self.tail = nn.SequentialCell(modules_tail)

    def construct(self, x):
        """rcan"""
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_pre_trained_param_dict(self, new_param_dict, strict=True):
        """
        load pre_trained param dict from rcan_x2
        """
        own_param = self.parameters_dict()
        for name, new_param in new_param_dict.items():
            if len(name) >= 4 and name[:4] == "net.":
                name = name[4:]
            if name in own_param:
                if isinstance(new_param, Parameter):
                    param = own_param[name]
                    if tuple(param.data.shape) == tuple(new_param.data.shape):
                        param.set_data(type(param.data)(new_param.data))
                    elif name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_param[name].shape, new_param.shape))
                elif strict:
                    if name.find('tail') == -1:
                        raise KeyError('unexpected key "{}" in parameters_dict()'
                                       .format(name))
