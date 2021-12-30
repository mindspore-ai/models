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
# =======================================================================================
""" yolox pa-fpn module """
import mindspore.nn as nn
from mindspore.ops import operations as P

from src.darknet import CSPDarknet
from src.network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Cell):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model
    """

    def __init__(
            self,
            input_w,
            input_h,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=None,
            depthwise=False,
            act="silu"
    ):
        super(YOLOPAFPN, self).__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024]
        self.input_w = input_w
        self.input_h = input_h
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample0 = P.ResizeNearestNeighbor((input_h // 16, input_w // 16))
        self.upsample1 = P.ResizeNearestNeighbor((input_h // 8, input_w // 8))
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )
        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.concat = P.Concat(axis=1)

    def construct(self, inputs):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        x2, x1, x0 = self.backbone(inputs)
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512  /32
        f_out0 = self.upsample0(fpn_out0)  # 512    /16
        f_out0 = self.concat((f_out0, x1))  # 512->1024    /16
        f_out0 = self.C3_p4(f_out0)  # 1024->512  /16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256  /16
        f_out1 = self.upsample1(fpn_out1)  # 256  /8
        f_out1 = self.concat((f_out1, x2))  # 256->512  /8
        pan_out2 = self.C3_p3(f_out1)  # 512->256  /16

        p_out1 = self.bu_conv2(pan_out2)  # 256->256  /16
        p_out1 = self.concat((p_out1, fpn_out1))  # 256->512  /16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = self.concat((p_out0, fpn_out0))  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        return pan_out2, pan_out1, pan_out0
