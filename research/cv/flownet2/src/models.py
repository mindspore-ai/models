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

from .submodels import FlowNetC
from .submodels import FlowNetS
from .submodels import FlowNetSD
from .submodels import FlowNetFusion
from .submodels.custom_ops.custom_ops import Resample2D as Resample2d
from .submodels.submodules import ChannelNorm
from .submodels.submodules import Upsample

Parameter_count = 162, 518, 834

class FlowNet2(nn.Cell):

    def __init__(self, rgb_max=255, batchNorm=False, div_flow=20.):
        super(FlowNet2, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max

        self.channelnorm = ChannelNorm(axis=1)

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(batchNorm=self.batchNorm)

        self.upsample1 = Upsample(scale_factor=4, mode='bilinear')

        self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = Upsample(scale_factor=4, mode='bilinear')
        self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(batchNorm=self.batchNorm)

        self.upsample3 = Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = Upsample(scale_factor=4, mode='nearest')

        self.resample3 = Resample2d()

        self.resample4 = Resample2d()

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(batchNorm=self.batchNorm)

        self.concat_op = ops.Concat(1)
        self.mean = ops.ReduceMean()

        for c in self.cells():
            if isinstance(c, nn.Conv2d):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'

            if isinstance(c, nn.Conv2dTranspose):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'


    def construct(self, inputs):
        rgb_mean = inputs.view(inputs.shape[:2] + (-1,)).mean(axis=-1).view(inputs.shape[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]

        x = self.concat_op((x1, x2))

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = self.concat_op((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0))

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = self.concat_op((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0))

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)

        diff_flownets2_img1 = self.channelnorm((x[:, :3, :, :] - diff_flownets2_flow))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)

        diff_flownetsd_img1 = self.channelnorm((x[:, :3, :, :] - diff_flownetsd_flow))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = self.concat_op(
            (x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow,
             diff_flownetsd_img1, diff_flownets2_img1))
        flownetfusion_flow = self.flownetfusion(concat3)

        return flownetfusion_flow


class FlowNet2C(FlowNetC.FlowNetC):
    def __init__(self, rgb_max, batchNorm=False, div_flow=20):
        super(FlowNet2C, self).__init__(batchNorm=batchNorm, div_flow=div_flow)
        self.rgb_max = rgb_max
        self.concat_op = ops.Concat(1)
        self.mean = ops.ReduceMean()

    def construct(self, inputs):
        rgb_mean = self.mean(inputs.view(inputs.shape[:2] + (-1,)), -1).view(inputs.shape[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

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
        return self.upsample1(flow2 * self.div_flow)


class FlowNet2S(FlowNetS.FlowNetS):
    def __init__(self, rgb_max=255, batchNorm=False, div_flow=20):
        super(FlowNet2S, self).__init__(input_channels=6, batchNorm=batchNorm)
        self.rgb_max = rgb_max
        self.div_flow = div_flow
        self.concat_op = ops.Concat(1)
        self.mean = ops.ReduceMean()

    def construct(self, inputs):
        rgb_mean = self.mean(inputs.view(inputs.shape[:2] + (-1,)), -1).view(inputs.shape[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = self.concat_op((x[:, :, 0, :, :], x[:, :, 1, :, :]))

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
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

        concat3 = self.concat_op((out_conv3, out_deconv3, flow4_up))
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = self.concat_op((out_conv2, out_deconv2, flow3_up))
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return self.upsample1(flow2 * self.div_flow)


class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, rgb_max=255, batchNorm=False, div_flow=20):
        super(FlowNet2SD, self).__init__(batchNorm=batchNorm)
        self.rgb_max = rgb_max
        self.div_flow = div_flow
        self.concat_op = ops.Concat(1)
        self.mean = ops.ReduceMean()

    def construct(self, inputs):
        rgb_mean = self.mean(inputs.view(inputs.shape[:2] + (-1,)), -1).view(inputs.shape[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = self.concat_op((x[:, :, 0, :, :], x[:, :, 1, :, :]))

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
        return self.upsample1(flow2 * self.div_flow)


class FlowNet2CS(nn.Cell):

    def __init__(self, rgb_max=255, batchNorm=False, div_flow=20.):
        super(FlowNet2CS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max

        self.channelnorm = ChannelNorm(axis=1)

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(batchNorm=self.batchNorm)
        self.upsample1 = Upsample(scale_factor=4, mode='bilinear')

        self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = Upsample(scale_factor=4, mode='bilinear')

        self.concat_op = ops.Concat(1)
        self.mean = ops.ReduceMean()

        for c in self.cells():
            if isinstance(c, nn.Conv2d):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'

            if isinstance(c, nn.Conv2dTranspose):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'

    def construct(self, inputs):
        rgb_mean = self.mean(inputs.view(inputs.shape[:2] + (-1,)), -1).view(inputs.shape[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = self.concat_op((x1, x2))

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = self.concat_op((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0))

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        return flownets1_flow


class FlowNet2CSS(nn.Cell):

    def __init__(self, rgb_max=255, batchNorm=False, div_flow=20.):
        super(FlowNet2CSS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max

        self.channelnorm = ChannelNorm(axis=1)

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(batchNorm=self.batchNorm)
        self.upsample1 = Upsample(scale_factor=4, mode='bilinear')

        self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = Upsample(scale_factor=4, mode='bilinear')

        self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample3 = Upsample(scale_factor=4, mode='nearest')

        self.concat_op = ops.Concat(1)
        self.mean = ops.ReduceMean()

        for c in self.cells():
            if isinstance(c, nn.Conv2d):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'

            if isinstance(c, nn.Conv2dTranspose):
                if c.bias_init is not None:
                    c.bias_init = 'Uniform'
                c.weight_init = 'XavierUniform'


    def construct(self, inputs):
        rgb_mean = self.mean(inputs.view(inputs.shape[:2] + (-1,)), -1).view(inputs.shape[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = self.concat_op((x1, x2))

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = self.concat_op((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0))

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = self.concat_op((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0))

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow
