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
from .submodules import Upsample

Parameter_count = 45, 371, 666


class FlowNetSD(nn.Cell):
    def __init__(self, batchNorm=True):
        super(FlowNetSD, self).__init__()

        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 6, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.inter_conv5 = i_conv(self.batchNorm, 1026, 512)
        self.inter_conv4 = i_conv(self.batchNorm, 770, 256)
        self.inter_conv3 = i_conv(self.batchNorm, 386, 128)
        self.inter_conv2 = i_conv(self.batchNorm, 194, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.Conv2dTranspose(2, 2, 4, 2, pad_mode='pad', padding=1, has_bias=True)
        self.upsampled_flow5_to_4 = nn.Conv2dTranspose(2, 2, 4, 2, pad_mode='pad', padding=1, has_bias=True)
        self.upsampled_flow4_to_3 = nn.Conv2dTranspose(2, 2, 4, 2, pad_mode='pad', padding=1, has_bias=True)
        self.upsampled_flow3_to_2 = nn.Conv2dTranspose(2, 2, 4, 2, pad_mode='pad', padding=1, has_bias=True)

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

        self.upsample1 = Upsample(scale_factor=4, mode='bilinear')

    def construct(self, x):
        # print(x.shape)
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = self.concat_op((out_conv5, out_deconv5, flow6_up))
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = self.concat_op((out_conv4, out_deconv4, flow5_up))
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = self.concat_op((out_conv3, out_deconv3, flow4_up))
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = self.concat_op((out_conv2, out_deconv2, flow3_up))
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return flow2, None
