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

import mindspore
import mindspore.nn as nn
import mindspore.ops as P

from src.pwc_modules import upsample2d_as
from src.pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense


class PWCNet(nn.Cell):
    '''define pwcnet work'''
    def __init__(self, div_flow=0.05):
        super(PWCNet, self).__init__()
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.pad = P.Pad(((0, 0), (self.search_range, self.search_range), (self.search_range, self.search_range),
                          (0, 0)))
        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.concat1 = P.Concat(3)
        self.concat2 = P.Concat(1)
        self.zeros = P.Zeros()
        self.transpose = P.Transpose()

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer(warp_type='bilinear')

        self.flow_estimators = nn.CellList()
        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr
            else:
                num_ch_in = self.dim_corr + ch + 2

            layer = FlowEstimatorDense(num_ch_in)
            self.flow_estimators.append(layer)

        self.context_networks = ContextNetwork(self.dim_corr + 32 + 2 + 448 + 2)

    def cost_volume(self, x1, x2_warp):
        padded_lvl = self.pad(x2_warp)
        _, h, w, _ = self.shape(x1)
        max_offset = self.search_range * 2 + 1

        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                slice_ = padded_lvl[:, y : y + h, x : x + w, :]
                cost = self.reduce_mean(x1 * slice_, 3)
                cost_vol.append(cost)
        cost_vol = self.concat1(cost_vol)
        return cost_vol

    def construct(self, x1_raw, x2_raw, training=True):

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = self.shape(x1_pyramid[0])
        flow = self.zeros((b_size, 2, h_x1, w_x1), mindspore.float32)

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = upsample2d_as(flow, x1)
                x2_warp = self.warping_layer(x2, flow)

            x1_ = self.transpose(x1, (0, 2, 3, 1))
            x2_warp = self.transpose(x2_warp, (0, 2, 3, 1))
            out_corr = self.cost_volume(x1_, x2_warp)
            out_corr = self.transpose(out_corr, (0, 3, 1, 2))
            out_corr_relu = self.leakyRELU(out_corr)

            # flow estimator
            if l == 0:
                x_intm, flow = self.flow_estimators[l](out_corr_relu)
            else:
                x_intm, flow = self.flow_estimators[l](self.concat2((out_corr_relu, x1, flow)))

            # upsampling or post-processing
            if l != self.output_level:
                flows.append(flow)
            else:
                flow_res = self.context_networks(self.concat2((x_intm, flow)))
                flow = flow + flow_res
                flows.append(flow)
                break


        if  training:
            return flows
        out_flow = upsample2d_as(flow, x1_raw) * (1.0 / self._div_flow)
        return out_flow


class BuildTrainNetwork(nn.Cell):
    '''BuildTrainNetwork'''
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__(auto_prefix=False)
        self.network = network
        self.criterion = criterion

    def construct(self, x1_raw, x2_raw, target):
        result = self.network(x1_raw, x2_raw)
        loss = self.criterion(result, target)
        return loss
