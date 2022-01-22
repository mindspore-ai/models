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
"""PNASNet-Mobile model definition"""
from collections import OrderedDict

import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype

from mindspore import Tensor
from mindspore.nn.loss.loss import LossBase

class MaxPool(nn.Cell):
    """
    MaxPool: MaxPool2d with zero padding.
    """
    def __init__(self, kernel_size, stride=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.pad = zero_pad
        if self.pad:
            self.zero_pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode='same')

    def construct(self, x):
        if self.pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.pad:
            x = x[:, :, 1:, 1:]
        return x

class SeparableConv2d(nn.Cell):
    """
    SeparableConv2d: Separable convolutions consist of first performing
    a depthwise spatial convolution followed by a pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, dw_kernel_size, dw_stride,
                 dw_padding):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                          kernel_size=dw_kernel_size, stride=dw_stride,
                                          pad_mode='pad', padding=dw_padding,
                                          group=in_channels, has_bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=1, pad_mode='pad', has_bias=False)

    def construct(self, x):
        """ construct network """
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x

class BranchSeparables(nn.Cell):
    """
    BranchSeparables: ReLU + Zero_Pad (when zero_pad is True) +  SeparableConv2d + BatchNorm2d +
                      ReLU + SeparableConv2d + BatchNorm2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 stem_cell=False, zero_pad=False):
        super(BranchSeparables, self).__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels

        self.pad = zero_pad
        if self.pad:
            self.zero_pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))

        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels,
                                           kernel_size, dw_stride=stride,
                                           dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(num_features=middle_channels, eps=0.001, momentum=0.9)

        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels,
                                           kernel_size, dw_stride=1,
                                           dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9)

    def construct(self, x):
        """ construct network """
        x = self.relu_1(x)
        if self.pad:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.pad:
            x = x[:, :, 1:, 1:]
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x

class ReluConvBn(nn.Cell):
    """
    ReluConvBn: ReLU + Conv2d + BatchNorm2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReluConvBn, self).__init__()
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9)

    def construct(self, x):
        """ construct network """
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class FactorizedReduction(nn.Cell):
    """
    FactorizedReduction is used to reduce the spatial size
    of the left input of a cell approximately by a factor of 2.
    """
    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        self.relu = nn.ReLU()

        path_1 = OrderedDict([
            ('avgpool', nn.AvgPool2d(kernel_size=1, stride=2, pad_mode='valid')),
            ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1,
                               pad_mode='pad', has_bias=False)),
        ])
        self.path_1 = nn.SequentialCell(path_1)

        self.path_2 = nn.CellList([])
        self.path_2.append(nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT"))
        self.path_2.append(
            nn.AvgPool2d(kernel_size=1, stride=2, pad_mode='valid')
        )
        self.path_2.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2 + int(out_channels % 2),
                      kernel_size=1, stride=1, pad_mode='pad', has_bias=False)
        )

        self.final_path_bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9)

    def construct(self, x):
        """ construct network """
        x = self.relu(x)
        x_path1 = self.path_1(x)

        x_path2 = self.path_2[0](x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2[1](x_path2)
        x_path2 = self.path_2[2](x_path2)

        out = self.final_path_bn(ops.Concat(1)((x_path1, x_path2)))
        return out

class CellBase(nn.Cell):
    """
    CellBase: PNASNet base unit.
    """
    def cell_forward(self, x_left, x_right):
        """
        cell_forward: to calculate the output according the x_left and x_right.
        """
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = ops.Concat(1)((x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))

        return x_out

class CellStem0(CellBase):
    """
    CellStemp0:PNASNet Stem0 unit
    """
    def __init__(self, in_channels_left, out_channels_left, in_channels_right,
                 out_channels_right):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right,
                                   kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(in_channels_left,
                                                 out_channels_left,
                                                 kernel_size=5, stride=2,
                                                 stem_cell=True)
        comb_iter_0_right = OrderedDict([
            ('max_pool', MaxPool(3, stride=2)),
            ('conv', nn.Conv2d(in_channels_left, out_channels_left,
                               kernel_size=1, has_bias=False)),
            ('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.9))
        ])
        self.comb_iter_0_right = nn.SequentialCell(comb_iter_0_right)

        self.comb_iter_1_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=7, stride=2)
        self.comb_iter_1_right = MaxPool(3, stride=2)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=5, stride=2)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
                                                  out_channels_right,
                                                  kernel_size=3, stride=2)
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=2)
        self.comb_iter_4_left = BranchSeparables(in_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3, stride=2,
                                                 stem_cell=True)
        self.comb_iter_4_right = ReluConvBn(out_channels_right,
                                            out_channels_right,
                                            kernel_size=1, stride=2)

    def construct(self, x_left):
        """ construct network """
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out

class Cell(CellBase):
    """
    Cell class that is used as a 'layer' in image architectures
    """
    def __init__(self, in_channels_left, out_channels_left, in_channels_right,
                 out_channels_right, is_reduction=False, zero_pad=False,
                 match_prev_layer_dimensions=False):
        super(Cell, self).__init__()

        stride = 2 if is_reduction else 1

        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left, out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left, out_channels_left, kernel_size=1)

        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left,
                                                 out_channels_left,
                                                 kernel_size=5, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=7, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=5, stride=stride,
                                                 zero_pad=zero_pad)
        self.comb_iter_2_right = BranchSeparables(out_channels_right,
                                                  out_channels_right,
                                                  kernel_size=3, stride=stride,
                                                  zero_pad=zero_pad)
        self.comb_iter_3_left = BranchSeparables(out_channels_right,
                                                 out_channels_right,
                                                 kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left,
                                                 out_channels_left,
                                                 kernel_size=3, stride=stride,
                                                 zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right,
                                                out_channels_right,
                                                kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def construct(self, x_left, x_right):
        """ construct network """
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out

class AuxLogits(nn.Cell):
    """
    AuxLogits: an auxiliary classifier to improve the convergence of very deep networks.
    """
    def __init__(self, in_channels, out_channels, name=None):
        super(AuxLogits, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(5, stride=3, pad_mode='valid')
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)

        self.conv_1 = nn.Conv2d(128, 768, (4, 4), pad_mode='valid')
        self.bn_1 = nn.BatchNorm2d(768)
        self.flatten = nn.Flatten()
        if name == 'large':
            self.fc = nn.Dense(6912, out_channels)  # large: 6912, mobile:768
        else:
            self.fc = nn.Dense(768, out_channels)

    def construct(self, x):
        """ construct network """
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class PNASNet5_Mobile(nn.Cell):
    """
    Progressive Neural Architecture Search (PNAS).
    Reference:
        Chenxi Liu et al. Progressive Neural Architecture SearchLearning Transferable Architectures.
                          ECCV 2018.
    Args:
        num_classes(int): The number of classes.
        enable_aux_logits(bool): whether to enable aux_logits. True for training, False(default) for evaluation.
        enable_dropout(bool): whether to enable dropout. True for training. False(default) for evaluation.
    Returns:
        Tensor, Tensor: the logits, aux_logits when enable_aux_logits is True.
        Tensor: the logits when enable_aux_logits is False.
    """

    def __init__(self, num_classes=1000, enable_aux_logits=False, enable_dropout=False):
        super(PNASNet5_Mobile, self).__init__()
        self.num_classes = num_classes
        self.enable_aux_logits = enable_aux_logits
        self.enable_dropout = enable_dropout

        self.conv_0 = nn.SequentialCell(OrderedDict([
            ('conv', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2,
                               pad_mode='pad', has_bias=False)),
            ('bn', nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.9))
        ]))

        self.cell_stem_0 = CellStem0(in_channels_left=32, out_channels_left=13,
                                     in_channels_right=32, out_channels_right=13)

        self.cell_stem_1 = Cell(in_channels_left=32, out_channels_left=27,
                                in_channels_right=65, out_channels_right=27,
                                match_prev_layer_dimensions=True,
                                is_reduction=True)
        self.cell_0 = Cell(in_channels_left=65, out_channels_left=54,
                           in_channels_right=135, out_channels_right=54,
                           match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=135, out_channels_left=54,
                           in_channels_right=270, out_channels_right=54)
        self.cell_2 = Cell(in_channels_left=270, out_channels_left=54,
                           in_channels_right=270, out_channels_right=54)
        self.cell_3 = Cell(in_channels_left=270, out_channels_left=108,
                           in_channels_right=270, out_channels_right=108,
                           is_reduction=True, zero_pad=True)
        self.cell_4 = Cell(in_channels_left=270, out_channels_left=108,
                           in_channels_right=540, out_channels_right=108,
                           match_prev_layer_dimensions=True)

        self.cell_5 = Cell(in_channels_left=540, out_channels_left=108,
                           in_channels_right=540, out_channels_right=108)

        if enable_aux_logits:
            self.aux_logits = AuxLogits(540, num_classes)
        else:
            self.aux_logits = None

        self.cell_6 = Cell(in_channels_left=540, out_channels_left=216,
                           in_channels_right=540, out_channels_right=216,
                           is_reduction=True)
        self.cell_7 = Cell(in_channels_left=540, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216,
                           match_prev_layer_dimensions=True)
        self.cell_8 = Cell(in_channels_left=1080, out_channels_left=216,
                           in_channels_right=1080, out_channels_right=216)
        self.relu = nn.ReLU()

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')

        if enable_dropout:
            self.dropout = nn.Dropout(keep_prob=0.5)
        else:
            self.dropout = None

        self.last_linear = nn.Dense(in_channels=1080, out_channels=num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        _initialize_weights: to initialize the weights.
        """
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2./n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    def features(self, x):
        """
        features: to calculate the features
        """
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)

        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)

        if self.enable_aux_logits:
            y_aux_logits = self.aux_logits(x_cell_5)
        else:
            y_aux_logits = None

        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)

        return x_cell_8, y_aux_logits

    def logits(self, features):
        """
        logits: to classify according to the features
        """
        y = self.relu(features)

        y = self.avg_pool(y)

        y = ops.Reshape()(y, (ops.Shape()(y)[0], -1,))

        if self.enable_dropout:
            y = self.dropout(y)

        y = self.last_linear(y)
        return y

    def construct(self, x):
        """ construct network """
        y, y_aux_logits = self.features(x)

        y = self.logits(y)

        if self.enable_aux_logits:
            return y, y_aux_logits
        return y

class CrossEntropy(LossBase):
    """
    CrossEntropy: the redefined loss function with SoftmaxCrossEntropyWithLogits
    """
    def __init__(self, smooth_factor=0, num_classes=1000, factor=0.4):
        super(CrossEntropy, self).__init__()
        self.factor = factor
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean(False)

    def construct(self, logit, aux, label):
        one_hot_label = self.onehot(label, ops.Shape()(logit)[1], self.on_value, self.off_value)
        loss_logit = self.ce(logit, one_hot_label)
        loss_logit = self.mean(loss_logit, 0)
        one_hot_label_aux = self.onehot(label, ops.Shape()(aux)[1], self.on_value, self.off_value)
        loss_aux = self.ce(aux, one_hot_label_aux)
        loss_aux = self.mean(loss_aux, 0)

        return loss_logit + self.factor*loss_aux

class PNASNet5_Mobile_WithLoss(nn.Cell):
    """
    Provide  pnasnet-mobile training loss through network.
    Args:
        config(dict): the config of pnasnet-mobile.
    Returns:
        Tensor: the loss of the network.
    """
    def __init__(self, config):
        super(PNASNet5_Mobile_WithLoss, self).__init__()
        self.network = PNASNet5_Mobile(num_classes=config.num_classes, enable_aux_logits=True, enable_dropout=True)
        self.loss = CrossEntropy(smooth_factor=0, num_classes=config.num_classes, factor=config.aux_factor)
        if config.device_target == 'GPU':
            self.loss = CrossEntropy(smooth_factor=config.label_smooth_factor,
                                     num_classes=config.num_classes, factor=config.aux_factor)
        self.cast = ops.Cast()

    def construct(self, data, label):
        logits, logits_aux = self.network(data)
        total_loss = self.loss(logits, logits_aux, label)
        return self.cast(total_loss, mstype.float32)
