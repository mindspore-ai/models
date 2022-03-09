# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The PDarts model file."""
import mindspore.nn as nn
import mindspore.ops as P

from src.operations import FactorizedReduce, ReLUConvBN, OPS
from src.my_utils import drop_path


class Module(nn.Cell):
    """
    The module of PDarts.
    """

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Module, self).__init__()
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
        self.div = P.Div()
        self.mul = P.Mul()
        self.concat_1 = P.Concat(axis=1)
        self.concat_start = self._concat.start
        self.concat_end = self.concat_start + len(self._concat) - 1

    def _compile(self, C, op_names, indices, concat, reduction):
        """
        Combine the functions of model.
        """
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

    def construct(self, s0, s1, drop_prob, layer_mask):
        """
        Do the module.
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        concat_result = None

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                h1 = drop_path(self.div, self.mul, h1,
                               drop_prob, layer_mask[i * 2])
                h2 = drop_path(self.div, self.mul, h2,
                               drop_prob, layer_mask[i * 2 + 1])
            s = h1 + h2
            states.append(s)
            if len(states) - 1 == self.concat_start + 1 and len(states) - 1 <= self.concat_end:
                concat_result = self.concat_1(
                    (states[len(states) - 2], states[len(states) - 1]))
            elif len(states) - 1 > self.concat_start + 1 and len(states) - 1 <= self.concat_end:
                concat_result = self.concat_1(
                    (concat_result, states[len(states) - 1]))

        return concat_result


class AuxiliaryHeadCIFAR(nn.Cell):
    """
    Define the Auxiliary Head.
    """

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.SequentialCell(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=3),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.reshape = P.Reshape()
        self.classifier = nn.Dense(768, num_classes)

    def construct(self, x):
        x = self.features(x)
        x = self.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x


class NetworkCIFAR(nn.Cell):
    """
    The PDarts model define
    """

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.SequentialCell(
            nn.Conv2d(3, C_curr, 3, padding=1, pad_mode='pad', has_bias=False),
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
            cell = Module(genotype, C_prev_prev, C_prev,
                          C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cell_list += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(
                C_to_auxiliary, num_classes)
        self.global_pooling = P.ReduceMean(keep_dims=True)
        self.reshape = P.Reshape()
        self.classifier = nn.Dense(C_prev, num_classes)

    def construct(self, x):
        """
        Do the model.
        """
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i in range(len(self.cell_list)):
            cell = self.cell_list[i]
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob, self.epoch_mask[i])
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1, (2, 3))
        out = self.reshape(out, (out.shape[0], -1))
        logits = self.classifier(out)
        if self._auxiliary and self.training:
            return logits, logits_aux
        return logits
