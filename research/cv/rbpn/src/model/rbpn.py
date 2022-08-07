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


"""rbpn"""

import mindspore
import mindspore.nn as nn
from src.model.base_networks import ConvBlock, ResnetBlock, DeconvBlock
from src.model.dbpns import Net as DBPNS



class Net(nn.Cell):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        # base_filter=256
        # feat=64
        self.nFrames = nFrames

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, num_stages, scale_factor)

        # Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.SequentialCell(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.SequentialCell(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.SequentialCell(*modules_body3)

        # Reconstruction
        self.output = ConvBlock((nFrames - 1) * feat, num_channels, 3, 1, 1, activation=None, norm=None)
        self.op = mindspore.ops.Concat(1)

    def construct(self, x, neighbor, flow):

        neighbor = neighbor.transpose((1, 0, 2, 3, 4))
        flow = flow.transpose((1, 0, 2, 3, 4))

        ### initial feature extraction
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(6):
            concat_xn = self.op((x, neighbor[j]))
            concat_xnf = self.op((concat_xn, flow[j]))
            feat_frame.append(self.feat1(concat_xnf))
        ####Projection
        Ht = []
        for j in range(6):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])
            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        ####Reconstruction
        concat_ht = Ht[0]
        for i in range(5):
            concat_ht = self.op((concat_ht, Ht[i + 1]))

        output = self.output(concat_ht)
        return output
