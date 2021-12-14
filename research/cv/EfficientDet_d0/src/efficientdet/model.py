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
""" define efficientdet components"""
import mindspore.nn as nn
from mindspore import ops as op
from mindspore import Parameter
import mindspore
from src.efficientnet.utils import MemoryEfficientSwish, Swish
from src.efficientnet.model import EfficientNet as EffNet

class SeparableConvBlock(nn.Cell):
    """ SeparableConvBlock """
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels


        self.depthwise_conv = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                        pad_mode="same", has_bias=False, group=in_channels)

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                        pad_mode="same", has_bias=True)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3, momentum=0.99)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.reducemean = op.ReduceMean(False)

    def construct(self, x):
        """ forward """
        x1 = self.depthwise_conv(x)
        x2 = self.pointwise_conv(x1)
        if self.norm:
            x2 = self.bn(x2)

        if self.activation:
            x2 = self.swish(x2)

        return x2


class BiFPN(nn.Cell):
    """ bifpn """
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=False):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """

        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = op.ResizeNearestNeighbor(size=(8, 8))
        self.p5_upsample = op.ResizeNearestNeighbor(size=(16, 16))
        self.p4_upsample = op.ResizeNearestNeighbor(size=(32, 32))
        self.p3_upsample = op.ResizeNearestNeighbor(size=(64, 64))

        self.p4_downsample = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.p5_downsample = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.p6_downsample = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.p7_downsample = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.SequentialCell(
                [nn.Conv2d(conv_channels[2], num_channels, 1, 1, pad_mode="same", padding=0, has_bias=True),
                 nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.99)]
            )
            self.p4_down_channel = nn.SequentialCell(
                [nn.Conv2d(conv_channels[1], num_channels, 1, pad_mode="same", padding=0, has_bias=True),
                 nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.99)]
            )
            self.p3_down_channel = nn.SequentialCell(
                [nn.Conv2d(conv_channels[0], num_channels, 1, pad_mode="same", padding=0, has_bias=True),
                 nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.99)]
            )

            self.p5_to_p6 = nn.SequentialCell(
                [nn.Conv2d(conv_channels[2], num_channels, 1, pad_mode="same", padding=0, has_bias=True),
                 nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.99),
                 nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")]
            )
            self.p6_to_p7 = nn.SequentialCell(
                [nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")]
            )


            self.p4_down_channel_2 = nn.SequentialCell(
                [nn.Conv2d(conv_channels[1], num_channels, 1, pad_mode="same", padding=0, has_bias=True),
                 nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.99)]
            )
            self.p5_down_channel_2 = nn.SequentialCell(
                [nn.Conv2d(conv_channels[2], num_channels, 1, pad_mode="same", padding=0, has_bias=True),
                 nn.BatchNorm2d(num_channels, eps=1e-3, momentum=0.99)]
            )

        ones = op.Ones()
        self.p6_w1 = Parameter(ones(2, mindspore.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = Parameter(ones(2, mindspore.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = Parameter(ones(2, mindspore.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = Parameter(ones(2, mindspore.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = Parameter(ones(3, mindspore.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = Parameter(ones(3, mindspore.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = Parameter(ones(3, mindspore.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = Parameter(ones(2, mindspore.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention
        self.squeeze = op.Squeeze()
        self.redecesum = op.ReduceSum(False)

    def construct(self, inputs):
        """ bifpn forward """

        # """
        # illustration of a minimal bifpn unit
        #     P7_0 -------------------------> P7_2 -------->
        #        |-------------|                ↑
        #                      ↓                |
        #     P6_0 ---------> P6_1 ---------> P6_2 -------->
        #        |-------------|--------------↑ ↑
        #                      ↓                |
        #     P5_0 ---------> P5_1 ---------> P5_2 -------->
        #        |-------------|--------------↑ ↑
        #                      ↓                |
        #     P4_0 ---------> P4_1 ---------> P4_2 -------->
        #        |-------------|--------------↑ ↑
        #                      |--------------↓ |
        #     P3_0 -------------------------> P3_2 -------->
        # """

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._construct_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._construct(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _construct_fast_attention(self, inputs):
        """ pylint """

        if self.first_time:

            p3, p4, p5 = inputs

            p3_in = self.p3_down_channel(p3)

            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight_1 = p6_w1 / (self.redecesum(p6_w1, 0) + self.epsilon)

            p6_td = self.conv6_up(self.swish(
                self.squeeze(weight_1[0:1:1]) * p6_in + self.squeeze(weight_1[1:2:1]) * self.p6_upsample(p7_in))) #[1,64,4,4]

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight_2 = p5_w1 / (self.redecesum(p5_w1, 0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(
                self.squeeze(weight_2[0:1:1]) * p5_in_1 + self.squeeze(weight_2[1:2:1]) * self.p5_upsample(p6_td)))    #[1,64,8,8]

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight_3 = p4_w1 / (self.redecesum(p4_w1, 0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(
                self.squeeze(weight_3[0:1:1]) * p4_in_1 + self.squeeze(weight_3[1:2:1]) * self.p4_upsample(p5_td)))    #[1,64,16,16]

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight_4 = p3_w1 / (self.redecesum(p3_w1, 0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(
                self.squeeze(weight_4[0:1:1]) * p3_in + self.squeeze(weight_4[1:2:1]) * self.p3_upsample(p4_td)))    #[1,64,32,32]

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight_5 = p4_w2 / (self.redecesum(p4_w2, 0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(self.squeeze(weight_5[0:1:1]) * p4_in_2 + self.squeeze(weight_5[1:2:1]) * p4_td +
                           self.squeeze(weight_5[2:3:1]) * self.p4_downsample(p3_out)))

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight_6 = p5_w2 / (self.redecesum(p5_w2, 0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(self.squeeze(weight_6[0:1:1]) * p5_in_2 + self.squeeze(weight_6[1:2:1]) * p5_td +
                           self.squeeze(weight_6[2:3:1]) * self.p5_downsample(p4_out)))

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight_7 = p6_w2 / (self.redecesum(p6_w2, 0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(self.squeeze(weight_7[0:1:1]) * p6_in + self.squeeze(weight_7[1:2:1]) * p6_td +
                           self.squeeze(weight_7[2:3:1]) * self.p6_downsample(p5_out)))

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight_8 = p7_w2 / (self.redecesum(p7_w2, 0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(
                self.squeeze(weight_8[0:1:1]) * p7_in + self.squeeze(weight_8[1:2:1]) * self.p7_downsample(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            reduce_op = mindspore.ops.ReduceSum(False)

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (reduce_op(p6_w1, 0) + self.epsilon)
            p6_td = self.conv6_up(self.swish(
                self.squeeze(weight[0:1:1]) * p6_in + self.squeeze(weight[1:2:1]) * self.p6_upsample(p7_in)))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (reduce_op(p5_w1, 0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(
                self.squeeze(weight[0:1:1]) * p5_in + self.squeeze(weight[1:2:1]) * self.p5_upsample(p6_td)))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (reduce_op(p4_w1, 0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(
                self.squeeze(weight[0:1:1]) * p4_in + self.squeeze(weight[1:2:1]) * self.p4_upsample(p5_td)))

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (reduce_op(p3_w1, 0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(
                self.squeeze(weight[0:1:1]) * p3_in + self.squeeze(weight[1:2:1]) * self.p3_upsample(p4_td)))

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (reduce_op(p4_w2, 0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(self.squeeze(weight[0:1:1]) * p4_in + self.squeeze(weight[1:2:1]) * p4_td + self.squeeze(
                    weight[2:3:1]) * self.p4_downsample(p3_out)))

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (reduce_op(p5_w2, 0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(self.squeeze(weight[0:1:1]) * p5_in + self.squeeze(weight[1:2:1]) * p5_td + self.squeeze(
                    weight[2:3:1]) * self.p5_downsample(p4_out)))

            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (reduce_op(p6_w2, 0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(self.squeeze(weight[0:1:1]) * p6_in + self.squeeze(weight[1:2:1]) * p6_td + self.squeeze(
                    weight[2:3:1]) * self.p6_downsample(p5_out)))

            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (reduce_op(p7_w2, 0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(
                self.squeeze(weight[0:1:1]) * p7_in + self.squeeze(weight[1:2:1]) * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class Regressor(nn.Cell):
    """ regressor """
    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.conv_list = nn.CellList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.CellList(
            [nn.CellList([nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.99) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.transpose = op.Transpose()
        self.reshape = op.Reshape()
        self.concat = op.Concat(axis=1)

    def construct(self, inputs):
        """ regressor forward"""
        feats = ()
        for feat, bn_list in zip(inputs, self.bn_list):
            for _, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)

            feat = self.header(feat)
            feat = self.transpose(feat, (0, 2, 3, 1))
            feat = self.reshape(feat, (feat.shape[0], -1, 4))

            feats += (feat,)

        feats = self.concat(feats)

        return feats


class Classifier(nn.Cell):
    """ classifier """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors  # 9
        self.num_classes = num_classes  # 90
        self.num_layers = num_layers  # 3

        self.conv_list = nn.CellList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])

        self.bn_list = nn.CellList(
            [nn.CellList([nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.99) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.transpose = op.Transpose()
        self.reshape = op.Reshape()
        self.concat = op.Concat(axis=1)
        self.sigmoid = nn.Sigmoid()
        self.reducemean = op.ReduceMean(False)

    def construct(self, inputs):
        """ classifier forward """
        feats = ()
        for feat, bn_list in zip(inputs, self.bn_list):
            for _, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)     # [BS, 810, 64, 64]
            feat = self.transpose(feat, (0, 2, 3, 1))     #[ 2, 64, 64, 810]
            feat = self.reshape(feat, (feat.shape[0], feat.shape[1], feat.shape[2],
                                       self.num_anchors, self.num_classes))         # [BS, 64, 64, 9, 90]
            feat = self.reshape(feat, (feat.shape[0], -1, self.num_classes))
            feats += (feat,)
        feats = self.concat(feats)
        feats = self.sigmoid(feats)

        return feats


class EfficientNet(nn.Cell):
    """ effcient net """
    def __init__(self, is_training=False):
        super(EfficientNet, self).__init__()

        self.model = EffNet.from_name("efficientnet-b0", is_training)

    def construct(self, x):
        """ forward """
        c1 = self.model.conv_stem(x)
        c2 = self.model.bn0(c1)
        c3 = self.model.swish(c2)

        c11 = self.model.blocks[0](c3)
        c12 = self.model.blocks[1](c11)
        c13 = self.model.blocks[2](c12)
        c14 = self.model.blocks[3](c13)
        c15 = self.model.blocks[4](c14)

        c21 = self.model.blocks[5](c15)
        c22 = self.model.blocks[6](c21)
        c23 = self.model.blocks[7](c22)
        c24 = self.model.blocks[8](c23)
        c25 = self.model.blocks[9](c24)
        c26 = self.model.blocks[10](c25)

        c31 = self.model.blocks[11](c26)
        c32 = self.model.blocks[12](c31)
        c33 = self.model.blocks[13](c32)
        c34 = self.model.blocks[14](c33)
        c35 = self.model.blocks[15](c34)

        return c15, c26, c35
