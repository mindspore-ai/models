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

"""NASFPN."""

import mindspore.nn as nn
from src.merge_cells import GlobalPoolingCell, SumCell

class NASFPN(nn.Cell):
    """
        NASFPN architecture.

    Args:
        in_channels (List(int)): Number of input channels per scale.
        out_channels (int): Numbers of block in different layers.
        num_outs (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
    Returns:
        Tensor, output tensor per scale.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 stack_times=7,
                 start_level=0, # [c1, c2, c3, c4, c5] -> [0, 1, 2, 3, 4]
                 end_level=-1):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.stack_times = stack_times

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        # add lateral connections
        self.lateral_convs = nn.CellList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2dBnAct(
                in_channels[i],
                out_channels,
                1,
                has_bn=True,
                activation=None)
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.CellList()
        for i in range(extra_levels):
            extra_conv = nn.Conv2dBnAct(
                out_channels,
                out_channels,
                1,
                has_bn=True,
                activation=None)
            self.extra_downsamples.append(
                nn.SequentialCell([extra_conv, nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")]))

        # add NAS FPN connections
        self.fpn_stages = nn.CellList()
        for _ in range(self.stack_times):
            # gp(p6, p4) -> p4_1
            gp_64_4 = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels)
            # sum(p4_1, p4) -> p4_2
            sum_44_4 = SumCell(
                in_channels=out_channels,
                out_channels=out_channels)
            # sum(p4_2, p3) -> p3_out
            sum_43_3 = SumCell(
                in_channels=out_channels,
                out_channels=out_channels)
            # sum(p3_out, p4_2) -> p4_out
            sum_34_4 = SumCell(
                in_channels=out_channels,
                out_channels=out_channels)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            gp_43_5 = GlobalPoolingCell(with_out_conv=False)
            sum_55_5 = SumCell(
                in_channels=out_channels,
                out_channels=out_channels)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            gp_54_7 = GlobalPoolingCell(with_out_conv=False)
            sum_77_7 = SumCell(
                in_channels=out_channels,
                out_channels=out_channels)
            # gp(p7_out, p5_out) -> p6_out
            gp_75_6 = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels)
            stage = nn.CellList([gp_64_4, sum_44_4, sum_43_3, sum_34_4, gp_43_5, sum_55_5, gp_54_7, sum_77_7, gp_75_6])
            self.fpn_stages.append(stage)

    def construct(self, inputs):
        """Forward function."""
        # build P3-P5
        feats = []
        for i in range(self.start_level, self.backbone_end_level):
            feats.append(self.lateral_convs[i](inputs[i + self.start_level]))
        # build P6-P7 on top of P5
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage[0](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage[1](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage[2](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage[3](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage[4](p4, p3, out_size=p5.shape[-2:])
            p5 = stage[5](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage[6](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage[7](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage[8](p7, p5, out_size=p6.shape[-2:])

        return p3, p4, p5, p6, p7
