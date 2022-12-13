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

from operations import OPS, ReLUConvBN, Identity, FactorizedReduce
from utils import drop_path
import mindspore.nn as nn
import mindspore.ops as ops

class ChannelSELayer(nn.Cell):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Dense(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Dense(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, input_tensor):
        batch_size, num_channels, _, _ = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = ops.Mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class Cell(nn.Cell):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, SE_layer=False):
        super(Cell, self).__init__()
        self._SE_layer_flag = SE_layer

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        if SE_layer:
            self.SE_layer = ChannelSELayer(C * 4)
        self.myconcat = ops.Concat(axis=1)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.CellList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def construct(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        out = self.myconcat(([states[i] for i in self._concat]))
        if self._SE_layer_flag:
            out = self.SE_layer(out)
        return out


class AuxiliaryHeadCIFAR(nn.Cell):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.SequentialCell(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=3),
            nn.Conv2d(C, 128, 1, has_bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2, has_bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.classifier = nn.Dense(768, num_classes)

    def construct(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Cell):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2),
            nn.Conv2d(C, 128, 1, has_bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2, has_bias=False),
            nn.ReLU()
        )
        self.classifier = nn.Dense(768, num_classes)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.features(x)
        x = self.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x


class NetworkCIFAR(nn.Cell):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, drop_path_prob=0.3):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.SequentialCell(
            nn.Conv2d(3, C_curr, 3, pad_mode="pad", padding=1, has_bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cell_list = nn.CellList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cell_list += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = ops.AdaptiveAvgPool2D(1)
        self.classifier = nn.Dense(C_prev, num_classes)
        self.reshape = ops.Reshape()

    def construct(self, inputs):
        logits_aux = None
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cell_list):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        out = self.reshape(out, (out.shape[0], -1))
        logits = self.classifier(out)
        if self._auxiliary and self.training:
            return logits, logits_aux
        return logits

class NetworkImageNet(nn.Cell):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE_layer=False):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.SequentialCell(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(),
            nn.Conv2d(C // 2, C, 3, stride=2, pad_mode="pad", padding=1, has_bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(C, C, 3, stride=2, pad_mode="pad", padding=1, has_bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.CellList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE_layer)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Dense(C_prev, num_classes)

    def construct(self, inputs):
        logits_aux = None
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
