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

from mindspore import nn
import mindspore.ops as ops
from .submodules import conv
from .submodules import deconv
from .submodules import i_conv
from .submodules import predict_flow

Parameter_count = 581, 226


class FlowNetFusion(nn.Cell):
    def __init__(self, batchNorm=True):
        super(FlowNetFusion, self).__init__()

        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 11, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)

        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)

        self.inter_conv1 = i_conv(self.batchNorm, 162, 32)
        self.inter_conv0 = i_conv(self.batchNorm, 82, 16)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.Conv2dTranspose(2, 2, 4, 2, pad_mode='pad', padding=1, has_bias=True)
        self.upsampled_flow1_to_0 = nn.Conv2dTranspose(2, 2, 4, 2, pad_mode='pad', padding=1, has_bias=True)

        self.concat_op = ops.Concat(1)

        for c in self.cells():
            if isinstance(c, nn.Conv2d):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'

            if isinstance(c, nn.Conv2dTranspose):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'

    def construct(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)

        concat1 = self.concat_op((out_conv1, out_deconv1, flow2_up))
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = self.concat_op((out_conv0, out_deconv0, flow1_up))
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)

        return flow0
