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
# ===========================================================================
"""Encoder of Auto-DeepLab."""
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr

from ..modules.genotypes import PRIMITIVES
from ..modules.operations import OPS, ReLUConvBN
from ..modules.schedule_drop_path import DropPath
from ..modules.bn import NormLeakyReLU


@constexpr
def scale_dimension(dim, scale):
    """
    Scale dimension by the given scale value.

    Inputs:
        - dim (Int) - current dimension.
        - scale (Float) - scale value to apply.
    Outputs:
        - dimension after scale.
    """
    return int((dim - 1.0) * scale + 1.0) if dim % 2 == 1 else int(dim * scale)


class NASBaseCell(nn.Cell):
    r"""
    Basic cell of NAS.

    Inputs:
        - prev_prev_input (Tensor) - Output_0 of previous layer, Tensor of shape (N, C_0, H_0, W_0).
        - prev_input (Tensor) - Output_1 of previous layer, Tensor of shape (N, C_1, H_1, W_1).
    Outputs:
        - prev_input (Tensor) - Identify with the input 'prev_input', Tensor of shape (N, C_1, H_1, W_1).
        - concat_feature (Tensor) - Tensor of shape (N, C_2, H_2, W_2)
    """
    def __init__(self,
                 block_multiplier,
                 prev_prev_fmultiplier,
                 prev_filter_multiplier,
                 cell_arch,
                 layer_num,
                 total_layers,
                 filter_multiplier,
                 downup_sample,
                 args=None):
        super(NASBaseCell, self).__init__()

        self.cast = ops.Cast()
        self.scast = ops.ScalarCast()

        self.cell_arch = cell_arch
        # self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.block_multiplier = self.scast(block_multiplier, mindspore.int32)

        self.interpolate = nn.ResizeBilinear()
        self.pre_preprocess = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, 'pad', args.bn_momentum, args.bn_eps,
                                         args.affine, args.use_ABN, args.parallel)
        self.preprocess = ReLUConvBN(self.C_prev, self.C_out, 1, 1, 0, 'pad', args.bn_momentum, args.bn_eps,
                                     args.affine, args.use_ABN, args.parallel)

        self.add = ops.Add()
        self.cat = ops.Concat(1)
        self.sum = ops.ReduceSum()
        self.equal = ops.Equal()

        self.operations = nn.CellList()

        if downup_sample == 1:
            self.scale = 0.5
        else:
            if downup_sample == -1:
                self.scale = 2
            else:
                if downup_sample == 0:
                    self.scale = 1
                else:
                    raise ValueError('downup_sample should be 1, 0 or -1')

        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = nn.SequentialCell(
                OPS[primitive](self.C_out, 1, args.bn_momentum, args.bn_eps,
                               affine=args.affine, use_abn=args.use_ABN, parallel=args.parallel),
                DropPath(args.drop_path_keep_prob, layer_num, total_layers, args.total_iters)
            )
            self.operations.append(op)

    def construct(self, prev_prev_input, prev_input):
        """construct"""
        feature_size_h = scale_dimension(prev_input.shape[2], self.scale)
        feature_size_w = scale_dimension(prev_input.shape[3], self.scale)

        s1 = self.interpolate(prev_input, (feature_size_h, feature_size_w), None, True)
        s0 = self.interpolate(prev_prev_input, (s1.shape[2], s1.shape[3]), None, True)

        process_s0 = self.pre_preprocess(s0)
        process_s1 = self.preprocess(s1)

        states = [process_s0, process_s1]
        cache = ()

        for block in range(self.block_multiplier):
            index = block * 2
            branch1 = self.cell_arch[index][0]
            branch2 = self.cell_arch[index + 1][0]

            temp1 = self.operations[index](states[branch1])
            temp2 = self.operations[index + 1](states[branch2])

            temps = self.add(temp1, temp2)
            states.append(temps)
            cache += (temps,)

        concat_feature = self.cat(cache)
        return prev_input, concat_feature


class Encoder(nn.Cell):
    r"""
        Encoder of Auto-DeepLab.

        Link all NASBaseCell according to the given network architecture.

        Inputs:
            - x (Tensor) - Tensor of shape (N, C, H, W).
        Outputs:
            - last_output (Tensor) - Tensor of shape (N, C_0, H_0, W_0)
            - low_level_feature (Tensor) - Tensor of shape (N, C_1, H_1, W_1)
        """
    def __init__(self,
                 network_arch,
                 cell_arch,
                 total_layers=12,
                 filter_multiplier=20,
                 block_multiplier=5,
                 args=None):
        super(Encoder, self).__init__()

        self._total_layers = total_layers

        initial_channels = 64 if args.initial_fm is None else args.initial_fm

        self.stem0 = nn.SequentialCell(
            nn.Conv2d(3, initial_channels, 3, stride=2, pad_mode='pad', padding=1, weight_init='HeNormal'),
            NormLeakyReLU(initial_channels, args.bn_momentum, args.bn_eps, parallel=args.parallel)
        )
        self.stem1 = nn.SequentialCell(
            nn.Conv2d(initial_channels, initial_channels, 3, pad_mode='same', weight_init='HeNormal'),
            NormLeakyReLU(initial_channels, args.bn_momentum, args.bn_eps, parallel=args.parallel)
        )
        self.stem2 = nn.SequentialCell(
            nn.Conv2d(initial_channels, initial_channels * 2, 3, stride=2, pad_mode='pad', padding=1,
                      weight_init='HeNormal'),
            NormLeakyReLU(initial_channels * 2, args.bn_momentum, args.bn_eps, parallel=args.parallel)
        )

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.NASCells = nn.CellList([])

        # layer 0
        level_0 = network_arch[0]
        downup_sample_0 = level_0
        _NASCell_0 = NASBaseCell(block_multiplier,
                                 initial_channels / block_multiplier,
                                 initial_channels * 2 / block_multiplier,
                                 cell_arch,
                                 0,
                                 self._total_layers,
                                 filter_multiplier * filter_param_dict[level_0],
                                 downup_sample_0,
                                 args)
        self.NASCells.append(_NASCell_0)

        # layer 1
        level_1 = network_arch[1]
        prev_level_1 = network_arch[0]
        downup_sample_1 = level_1 - prev_level_1
        _NASCell_1 = NASBaseCell(block_multiplier,
                                 initial_channels * 2 / block_multiplier,
                                 filter_multiplier * filter_param_dict[prev_level_1],
                                 cell_arch,
                                 1,
                                 self._total_layers,
                                 filter_multiplier * filter_param_dict[level_1],
                                 downup_sample_1,
                                 args)
        self.NASCells.append(_NASCell_1)

        for i in range(2, self._total_layers):

            level = network_arch[i]
            prev_level = network_arch[i - 1]
            prev_prev_level = network_arch[i - 2]

            downup_sample = level - prev_level

            _NASCell_i = NASBaseCell(block_multiplier,
                                     filter_multiplier * filter_param_dict[prev_prev_level],
                                     filter_multiplier * filter_param_dict[prev_level],
                                     cell_arch,
                                     i,
                                     self._total_layers,
                                     filter_multiplier * filter_param_dict[level],
                                     downup_sample,
                                     args)

            self.NASCells.append(_NASCell_i)

    def construct(self, x):
        """construct"""
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)
        two_last_inputs_0 = (stem1, stem2)

        two_last_inputs_1 = self.NASCells[0](two_last_inputs_0[0], two_last_inputs_0[1])
        two_last_inputs_2 = self.NASCells[1](two_last_inputs_1[0], two_last_inputs_1[1])
        two_last_inputs_i = self.NASCells[2](two_last_inputs_2[0], two_last_inputs_2[1])
        low_level_feature = two_last_inputs_i[1]

        for i in range(3, self._total_layers):
            two_last_inputs_i = self.NASCells[i](two_last_inputs_i[0], two_last_inputs_i[1])

        last_output = two_last_inputs_i[-1]

        return last_output, low_level_feature


def get_default_arch():
    """Obtain default architecture of Encoder network and NasCell as the paper described."""
    backbone = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]

    cell = np.zeros((10, 2))
    cell[0] = [0, 7]
    cell[1] = [1, 4]
    cell[2] = [0, 4]
    cell[3] = [1, 6]
    cell[4] = [0, 4]
    cell[5] = [3, 4]
    cell[6] = [2, 5]
    cell[7] = [4, 5]
    cell[8] = [5, 7]
    cell[9] = [3, 5]
    cell = cell.astype('uint8')
    cell_arch = cell.tolist()

    return backbone, cell_arch
