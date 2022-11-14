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
# This file refers to the project https://github.com/MhLiao/DB.git

"""SegDetector, support DBNet & DBNet++"""

from mindspore import ops
import mindspore.nn as nn
from mindspore.common.initializer import HeNormal
from mindspore.common import initializer as init

from .asf import ScaleFeatureSelection


class SegDetector(nn.Cell):
    def __init__(self, in_channels, inner_channels=256, k=50,
                 bias=False, adaptive=True, serial=False, training=False, concat_attention=False,
                 attention_type='scale_channel_spatial'):
        '''
        in_channels: resnet18=[64, 128, 256, 512]
                    resnet50=[2048,1024,512,256]
        inner_channels: Inner channels in Conv2d
        k: The k to calculate binary graph.
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''

        super(SegDetector, self).__init__()

        self.k = k
        self.serial = serial
        self.training = training

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, has_bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, has_bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, has_bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, has_bias=bias)

        self.out5 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        self.out4 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        self.out3 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias)

        self.binarize = nn.SequentialCell(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(inner_channels // 4, inner_channels // 4, 2, stride=2, has_bias=True),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(inner_channels // 4, 1, 2, stride=2, has_bias=True),
            nn.Sigmoid())
        self.attention = concat_attention
        if self.attention:
            self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels // 4,
                                                          attention_type=attention_type)
            self.concat_attention.weights_init(self.concat_attention)

        self.weights_init(self.binarize)

        self.adaptive = adaptive

        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, bias=bias)
            self.weights_init(self.thresh)

        self.weights_init(self.in5)
        self.weights_init(self.in4)
        self.weights_init(self.in3)
        self.weights_init(self.in2)

        self.weights_init(self.out5)
        self.weights_init(self.out4)
        self.weights_init(self.out3)
        self.weights_init(self.out2)

    def weights_init(self, c):
        for m in c.cells():
            if isinstance(m, nn.Conv2dTranspose):
                m.weight = init.initializer(HeNormal(), m.weight.shape)
                m.bias = init.initializer('zeros', m.bias.shape)

            elif isinstance(m, nn.Conv2d):
                m.weight = init.initializer(HeNormal(), m.weight.shape)

            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def _init_thresh(self, inner_channels, serial=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.SequentialCell(
            nn.Conv2d(in_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, in_channels // 4, 2, stride=2, has_bias=True),
            # size * 2
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, 1, 2, stride=2, has_bias=True),
            nn.Sigmoid())

        return self.thresh

    def construct(self, features):

        # shapes for inference:
        # [1, 64, 184, 320]
        # [1, 128, 92, 160]
        # [1, 256, 46, 80]
        # [1, 512, 23, 40]

        c2, c3, c4, c5 = features

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        # Carry out up sampling and prepare for connection
        up5 = ops.ResizeNearestNeighbor((in4.shape[2], in4.shape[3]))
        up4 = ops.ResizeNearestNeighbor((in3.shape[2], in3.shape[3]))
        up3 = ops.ResizeNearestNeighbor((in2.shape[2], in2.shape[3]))

        out4 = up5(in5) + in4  # 1/16
        out3 = up4(out4) + in3  # 1/8
        out2 = up3(out3) + in2  # 1/4

        upsample = ops.ResizeNearestNeighbor((c2.shape[2], c2.shape[3]))

        # The connected results are upsampled to make them the same shape, 1/4
        p5 = upsample(self.out5(in5))
        p4 = upsample(self.out4(out4))
        p3 = upsample(self.out3(out3))
        p2 = upsample(self.out2(out2))

        fuse = ops.Concat(1)((p5, p4, p3, p2))
        if self.attention:
            fuse = self.concat_attention(fuse, [p5, p4, p3, p2])

        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)

        pred = {}
        pred['binary'] = binary

        if self.adaptive and self.training:
            thresh = self.thresh(fuse)

            pred['thresh'] = thresh
            pred['thresh_binary'] = self.step_function(binary, thresh)

        return pred

    def step_function(self, x, y):
        """Get the binary graph through binary and threshold."""
        reciprocal = ops.Reciprocal()
        exp = ops.Exp()

        return reciprocal(1 + exp(-self.k * (x - y)))
