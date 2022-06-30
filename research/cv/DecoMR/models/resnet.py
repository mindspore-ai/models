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

import mindspore as ms
import mindspore.nn as nn
from mindspore import load_checkpoint

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding="same", stride=stride, has_bias=False)


class Bottleneck(nn.Cell):
    """
    Bottleneck layer
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, pad_mode="pad", has_bias=False,
                               dilation=dilation_)

        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.get_parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        forword
        """
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
    """
    resnet
    """
    def __init__(self, block, layers):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
                               has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        make layer
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        layers = [block(self.in_planes, planes, stride, dilation_=dilation, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation_=dilation))

        return nn.SequentialCell(*layers)

    def load_pretrained_model(self, model_file):
        """
        load pretrained model
        """
        load_checkpoint(model_file, net=self)

    def construct(self, x):
        """
        forward
        """
        feature_list = []
        feature_list.append(x)

        x1 = self.conv1(x)

        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        feature_list.append(x1)

        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        feature_list.append(x2)

        x3 = self.layer2(x2)
        feature_list.append(x3)

        x4 = self.layer3(x3)
        feature_list.append(x4)

        x5 = self.layer4(x4)
        feature_list.append(x5)
        y = self.avgpool(x5)
        y = y.view(y.shape[0], -1)

        return y, feature_list

class AdaptiveAvgPool2D(nn.Cell):
    """
    AdaptiveAvgPool2D layer
    """
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.adaptive_avg_pool = ms.ops.AdaptiveAvgPool2D(output_size)

    def construct(self, x):
        """
        forward
        :param x:
        :return:
        """
        return self.adaptive_avg_pool(x)


class ResNetLocate(nn.Cell):
    """
    resnet for resnet101
    """
    def __init__(self, block, layers):
        super(ResNetLocate, self).__init__()
        self.resnet = ResNet(block, layers)
        self.in_planes = 512
        self.out_planes = [512, 256, 256, 128]

        self.ppms_pre = nn.Conv2d(2048, self.in_planes, 1, 1, bias=False)
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(
                nn.SequentialCell(AdaptiveAvgPool2D(ii),
                                  nn.Conv2d(self.in_planes, self.in_planes, 1, 1, has_bias=False),
                                  nn.ReLU()))
        self.ppms = nn.CellList(ppms)

        self.ppm_cat = nn.SequentialCell(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False),
                                         nn.ReLU())
        for ii in self.out_planes:
            infos.append(nn.SequentialCell(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU()))
        self.infos = nn.CellList(infos)

        self.resize_bilinear = nn.ResizeBilinear()
        self.cat = ms.ops.Concat(axis=1)

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model)

    def construct(self, x):
        """
        forward
        """
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(self.resize_bilinear(self.ppms[k](xs_1), xs_1.size()[2:], align_corners=True))
        xls = self.ppm_cat(self.cat(xls))
        top_score = None

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](
                self.resize_bilinear(xls, xs[len(self.infos) - 1 - k].size()[2:], align_corners=True)))

        return xs, top_score, infos


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    """
    model_file = "data/ms_resnet50.ckpt"
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_checkpoint(model_file, net=model)
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    """
    model_file = ""
    model = ResNetLocate(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        load_checkpoint(model_file, net=model)
    return model


if __name__ == "__main__":
    name = "resnet50"
    net = resnet50()
    num_params = 0
    num_layers = 0
    for n, param in net.parameters_and_names():
        if "moving_" in n:
            continue
        num_params += param.size
        num_layers += 1
    print(name)
    print(net)
    print(f"The number of layers: {num_layers}")
    print(f"The number of parameters: {num_params}")
