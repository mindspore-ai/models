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
"""
deeplab v3 with res2net as backbone
"""

import math
import mindspore.nn as nn
from mindspore.ops import operations as P


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, weight_init='xavier_uniform')


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=padding,
                     dilation=dilation, weight_init='xavier_uniform')


class Res2net(nn.Cell):
    """Res2net"""
    def __init__(self, block, block_num, output_stride, use_batch_statistics=True):
        super(Res2net, self).__init__()
        self.inplanes = 64
        self.conv1_0 = nn.Conv2d(3, 32, 3, stride=2, pad_mode='pad', padding=3,
                                 weight_init='xavier_uniform')
        self.bn1_0 = nn.BatchNorm2d(32, use_batch_statistics=use_batch_statistics)
        self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, pad_mode='pad', padding=3,
                                 weight_init='xavier_uniform')
        self.bn1_1 = nn.BatchNorm2d(32, use_batch_statistics=use_batch_statistics)
        self.conv1_2 = nn.Conv2d(32, self.inplanes, 3, stride=1, pad_mode='pad', padding=3,
                                 weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, block_num[0], use_batch_statistics=use_batch_statistics)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2, use_batch_statistics=use_batch_statistics)

        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=2, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=4, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None, use_batch_statistics=True):
        """_make_layer"""
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.SequentialCell([
                    nn.MaxPool2d(kernel_size=1, stride=1, pad_mode='same'),
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
                ])
            else:
                downsample = nn.SequentialCell([
                    nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
                ])

        if grids is None:
            grids = [1] * blocks

        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0],
                  use_batch_statistics=use_batch_statistics, stype='stage')
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=base_dilation * grids[i],
                      use_batch_statistics=use_batch_statistics))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        x = self.conv1_0(x)
        x = self.bn1_0(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        out = self.conv1_2(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Bottle2neck(nn.Cell):
    """Bottle2neck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_batch_statistics=True,
                 baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        assert scale > 1, "Res2Net is ResNet when scale = 1"
        width = int(math.floor(planes * self.expansion // self.expansion * (baseWidth / 64.0)))
        channel = width * scale
        self.conv1 = conv1x1(inplanes, channel)
        self.bn1 = nn.BatchNorm2d(channel, use_batch_statistics=use_batch_statistics)

        if stype == 'stage' and stride == 2:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, pad_mode="same")

        # self.convs = nn.CellList()
        # self.bns = nn.CellList()
        self.convs = []
        self.bns = []
        for _ in range(scale - 1):
            self.convs.append(conv3x3(width, width, stride, dilation, dilation))
            self.bns.append(nn.BatchNorm2d(width, use_batch_statistics=use_batch_statistics))
        self.convs = nn.CellList(self.convs)
        self.bns = nn.CellList(self.bns)
        # self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        # self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv3 = conv1x1(channel, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, use_batch_statistics=use_batch_statistics)

        self.relu = nn.ReLU()
        self.downsample = downsample

        self.add = P.Add()
        self.scale = scale
        self.width = width
        self.stride = stride
        self.stype = stype
        self.split = P.Split(axis=1, output_num=scale)
        self.cat = P.Concat(axis=1)

    def construct(self, x):
        """construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        spx = self.split(out)

        sp = self.convs[0](spx[0])
        sp = self.relu(self.bns[0](sp) * P.OnesLike()(sp))
        out = sp
        # print("this")
        for i in range(1, self.scale - 1):
            # print(out.shape)
            if self.stype == 'stage':
                sp = spx[i]
            else:
                # this might be a bug of ms1.0
                # sp = spx[i] # model can converge
                # sp = sp[:, :, :, :]
                sp = sp + spx[i]
                # sp = sp + spx[i]  # model cannot converge
                # sp = self.add(sp, spx[i]) # model cannot converge
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp) * P.OnesLike()(sp))
            out = self.cat((out, sp))

        if self.stride == 1:
            out = self.cat((out, spx[self.scale - 1]))
        elif self.stride == 2:
            out = self.cat((out, self.pool(spx[self.scale - 1])))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out

class ASPP(nn.Cell):
    """ASPP"""
    def __init__(self, atrous_rates, phase='train', in_channels=2048, num_classes=21,
                 use_batch_statistics=True):
        super(ASPP, self).__init__()
        self.phase = phase
        out_channels = 256
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        self.aspp_pooling = ASPPPooling(in_channels, out_channels, use_batch_statistics=use_batch_statistics)
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
                               weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, weight_init='xavier_uniform', has_bias=True)
        self.concat = P.Concat(axis=1)
        self.drop = nn.Dropout(p=0.7)

    def construct(self, x):
        """construct"""
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)

        x = self.concat((x1, x2))
        x = self.concat((x, x3))
        x = self.concat((x, x4))
        x = self.concat((x, x5))

        x = self.conv1(x)
        x = self.bn1(x) * P.OnesLike()(x)
        x = self.relu(x)
        if self.phase == 'train':
            x = self.drop(x)
        x = self.conv2(x)
        return x


class ASPPPooling(nn.Cell):
    """ASPPPooling"""
    def __init__(self, in_channels, out_channels, use_batch_statistics=True):
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='xavier_uniform'),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        out = nn.AvgPool2d(size[2])(x)
        out = self.conv(out)
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class ASPPConv(nn.Cell):
    """ASPPConv"""
    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='xavier_uniform')
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
                             dilation=atrous_rate, weight_init='xavier_uniform')
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def construct(self, x):
        out = self.aspp_conv(x)
        return out


class DeepLabV3(nn.Cell):
    """DeepLabV3"""
    def __init__(self, phase='train', num_classes=21, output_stride=16, freeze_bn=False):
        super(DeepLabV3, self).__init__()
        use_batch_statistics = not freeze_bn
        self.res2net = Res2net(Bottle2neck, [3, 4, 23, 3], output_stride=output_stride,
                               use_batch_statistics=use_batch_statistics)
        self.aspp = ASPP([1, 6, 12, 18], phase, 2048, num_classes,
                         use_batch_statistics=use_batch_statistics)
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        out = self.res2net(x)
        out = self.aspp(out)
        out = P.ResizeBilinear((size[2], size[3]), True)(out)
        return out
