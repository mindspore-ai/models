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
""" CPNet """
from src.model.resnet import resnet50
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import load_param_into_net, load_checkpoint


class ResNet(nn.Cell):
    """ The pretrained ResNet """

    def __init__(self, pretrained_path, pretrained=False, deep_base=False, BatchNorm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        resnet = resnet50()
        if pretrained:
            params = load_checkpoint(pretrained_path)
            load_param_into_net(resnet, params)
        self.layer1 = nn.SequentialCell(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4
        self.resize_ops = ops.ResizeBilinear(
            (60, 60), True
        )
    def construct(self, x):
        """ ResNet process """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_aux = self.layer4(x)
        x = self.resize_ops(self.layer5(x_aux))
        return x


class ConvBNReLU(nn.Cell):
    """ basic module """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, pad_mode='pad', **kwargs)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()

    def construct(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class Aggregation(nn.Cell):
    """Aggregation Module"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
    ):
        super(Aggregation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2

        self.reduce_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.t1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            pad_mode='pad',
            padding=(padding, padding, 0, 0),
            group=out_channels,
        )
        self.t2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            pad_mode='pad',
            padding=(0, 0, padding, padding),
            group=out_channels,
        )

        self.p1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            pad_mode='pad',
            padding=(0, 0, padding, padding),
            group=out_channels,
        )
        self.p2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            pad_mode='pad',
            padding=(padding, padding, 0, 0),
            group=out_channels,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        out = self.relu(self.norm(x1 + x2))
        return out


class AuxLayer(nn.Cell):
    """ aux layer """
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()
        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)
        self.dropout = nn.Dropout(p=1 - dropout_prob)
        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def construct(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class CPNet(nn.Cell):
    """ cp layer """
    def __init__(self,
                 prior_channels,
                 proir__size,
                 am_kernel_size,
                 pretrained=True,
                 groups=1,
                 pretrained_path="",
                 deep_base=False,
                 BatchNorm_layer=nn.BatchNorm2d
                 ):
        super().__init__()

        self.in_channels = 2048
        self.channels = 21
        self.backbone = ResNet(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            deep_base=deep_base,
            BatchNorm_layer=BatchNorm_layer)
        self.prior_channels = prior_channels
        self.prior_size = Tensor([proir__size, proir__size])
        self.aggregation = Aggregation(self.in_channels, prior_channels, am_kernel_size)
        self.prior_conv = nn.SequentialCell(
            nn.Conv2d(self.prior_channels, proir__size * proir__size, 1, padding=0, group=groups),
            nn.BatchNorm2d(proir__size * proir__size))
        self.intra_conv = ConvBNReLU(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
        )
        self.inter_conv = ConvBNReLU(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
        )
        self.bottleneck = ConvBNReLU(
            self.in_channels + self.prior_channels * 2,
            self.channels,
            3,
            padding=1,
        )
        self.resize_bilinear = nn.ResizeBilinear()
        self.sigmoid = ops.Sigmoid()
        self.bmm = ops.BatchMatMul()
        self.cat = ops.Concat(axis=1)
        self.aux = AuxLayer(
            in_channels=self.in_channels,
            out_channels=21, inter_channels=self.channels)

    def construct(self, inputs):
        """ construct cp layer """
        H = inputs.shape[2]
        W = inputs.shape[3]
        if H != 480 or W != 480:
            inputs = self.resize_bilinear(inputs, size=(480, 480), align_corners=True)
        conv4 = self.backbone(inputs)
        batch_size, _, height, width = conv4.shape
        value = self.aggregation(conv4)
        context_prior_map = self.prior_conv(value)
        p_size = self.prior_size[0] * self.prior_size[1]
        context_prior_map = context_prior_map.reshape((batch_size, height * width, height * width))
        context_prior_map = ops.transpose(context_prior_map, (0, 2, 1))
        context_prior_map = self.sigmoid(context_prior_map)
        inter_context_prior_map = 1 - context_prior_map
        value = value.reshape((batch_size, self.prior_channels, -1))
        value = ops.transpose(value, (0, 2, 1))
        intra_context = self.bmm(context_prior_map, value)
        intra_context = intra_context / p_size
        intra_context = intra_context.transpose((0, 2, 1))
        intra_context = intra_context.reshape((batch_size, self.prior_channels, height, width))
        intra_context = self.intra_conv(intra_context)
        inter_context = self.bmm(inter_context_prior_map, value)
        inter_context = inter_context / p_size
        inter_context = inter_context.transpose((0, 2, 1))
        inter_context = inter_context.reshape((batch_size, self.prior_channels, height, width))
        inter_context = self.inter_conv(inter_context)
        cp_outs = self.cat([conv4, intra_context, inter_context])
        output = self.bottleneck(cp_outs)
        output = self.resize_bilinear(output, (H, W), align_corners=True)
        aux = self.aux(conv4)
        aux = self.resize_bilinear(aux, size=(H, W), align_corners=True)
        return output, aux, context_prior_map
