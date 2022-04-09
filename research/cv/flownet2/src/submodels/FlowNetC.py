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

import mindspore.nn as nn
import mindspore.ops as ops
from .custom_ops.custom_ops import Correlation
from .submodules import conv
from .submodules import predict_flow
from .submodules import deconv
from .submodules import Upsample


Parameter_count = 39, 175, 298


class FlowNetC(nn.Cell):
    def __init__(self, batchNorm=True, div_flow=20):
        super(FlowNetC, self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2)

        self.corr_activation = nn.LeakyReLU(0.1)
        self.conv3_1 = conv(self.batchNorm, 473, 256)
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

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

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
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3::, :, :]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a,
                             out_conv3b)  # 未打印 Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2)
        out_corr = self.corr_activation(out_corr)  # nn.LeakyReLU(0.1)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)  # 已打印 conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        in_conv3_1 = self.concat_op((out_conv_redir, out_corr))

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = self.concat_op((out_conv5, out_deconv5, flow6_up))

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = self.concat_op((out_conv4, out_deconv4, flow5_up))

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = self.concat_op((out_conv3_1, out_deconv3, flow4_up))

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = self.concat_op((out_conv2a, out_deconv2, flow3_up))

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return flow2, None
