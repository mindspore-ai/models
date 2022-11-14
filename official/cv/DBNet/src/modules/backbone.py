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
# This file refers to the project https://github.com/MhLiao/DB.git

"""ResNet & ResNet-DCNv2"""

import math
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import Normal, HeNormal


# set initializer to constant for debugging.
def conv3x3(inplanes, outplanes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, pad_mode="pad",
                     padding=1, weight_init=HeNormal())


class ModulatedDeformConv2d(nn.Conv2d):
    """
    A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.
    Args:
        in_channels (int): Same as nn.Conv2d. The channel number of the input tensor of the Conv2d layer.
        out_channels (int): Same as nn.Conv2d. The channel number of the output tensor of the Conv2d layer.
        kernel_size (int or tuple[int]): Same as nn.Conv2d. Specifies the height and width of the 2D convolution kernel.
        stride (int): Same as nn.Conv2d, while tuple is not supported. Default: 1.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        has_bias (bool: Same as nn.Conv2d. False.
        """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, has_bias=False):
        super(ModulatedDeformConv2d, self).__init__(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    pad_mode="pad",
                                                    padding=padding,
                                                    has_bias=has_bias)
        self.deform_groups = 1
        self.de_stride = (1, 1, stride, stride)
        self.de_padding = (padding, padding, padding, padding)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad_mode="pad",
            padding=self.padding,
            dilation=self.dilation,
            has_bias=True,
            weight_init="Zero",
        )

    def construct(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = ops.split(out, axis=1, output_num=3)
        mask = ops.Sigmoid()(mask)
        offset = ops.concat((o1, o2, mask), axis=1)
        out = ops.deformable_conv2d(x, self.weight, offset, self.kernel_size, self.de_stride, self.de_padding,
                                    bias=self.bias, deformable_groups=self.deform_groups, modulated=True)
        return out


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        if dcn:
            self.conv2 = ModulatedDeformConv2d(planes, planes, kernel_size=3, padding=1)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if dcn:
            self.conv2 = ModulatedDeformConv2d(planes, planes, kernel_size=3, padding=1, stride=stride)
        else:
            self.conv2 = conv3x3(planes, planes, stride=stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()

        self.downsample = downsample

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, dcn=False):

        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
                               has_bias=False, weight_init=HeNormal())
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.SequentialCell([
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
            nn.MaxPool2d(kernel_size=3, stride=2)])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_init = Normal(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample的功能是调整residual使其和out保持相同尺寸，out的变化由plane和stride控制
            downsample = nn.SequentialCell(
                # set initializer to constant for debugging.
                nn.Conv2d(self.inplanes, planes * block.expansion, pad_mode="pad",
                          kernel_size=1, stride=stride, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5


def resnet18(pretrained=True, backbone_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        ms_dict = ms.load_checkpoint(backbone_ckpt)
        param_not_load = ms.load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def deformable_resnet18(pretrained=True, backbone_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=True, **kwargs)

    if pretrained:
        ms_dict = ms.load_checkpoint(backbone_ckpt)
        param_not_load = ms.load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def resnet50(pretrained=True, backbone_ckpt=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        ms_dict = ms.load_checkpoint(backbone_ckpt)
        param_not_load = ms.load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def deformable_resnet50(pretrained=True, backbone_ckpt=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=True, **kwargs)

    if pretrained:
        ms_dict = ms.load_checkpoint(backbone_ckpt)
        param_not_load = ms.load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def get_backbone(initializer):
    backbone_dict = {
        "resnet18": resnet18,
        "deformable_resnet18": deformable_resnet18,
        "resnet50": resnet50,
        "deformable_resnet50": deformable_resnet50,
    }
    return backbone_dict[initializer]
