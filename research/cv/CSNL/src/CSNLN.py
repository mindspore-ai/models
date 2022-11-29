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
from mindspore import nn
import mindspore.ops as ops
from src.Attention import CrossScaleAttention, NonLocalAttention
from src.public import ResBlock, default_conv, BasicBlock, MeanShift
from model_utils.config import config


def make_model(args):
    return CSNLN(args)


class MultiSourceProjection(nn.Cell):
    def __init__(self, in_channel, kernel_size=3, scale=2, conv=default_conv):
        super(MultiSourceProjection, self).__init__()
        deconv_ksize, stride, padding, up_factor = {
            2: (6, 2, 2, 2),
            3: (9, 3, 3, 3),
            4: (6, 2, 2, 2)
        }[scale]
        self.up_attention = CrossScaleAttention(scale=up_factor)
        self.down_attention = NonLocalAttention()
        self.upsample = nn.SequentialCell(
            *[nn.Conv2dTranspose(in_channel, in_channel, deconv_ksize, stride=stride, padding=padding, pad_mode='pad'),
              nn.PReLU(in_channel)]
        )
        self.encoder = ResBlock(conv, in_channel, kernel_size, act=nn.PReLU(in_channel), res_scale=1)

    def construct(self, x):
        down_map = self.upsample(self.down_attention(x))
        up_map = self.up_attention(x)

        err = self.encoder(up_map - down_map)
        final_map = down_map + err

        return final_map


class RecurrentProjection(nn.Cell):
    def __init__(self, in_channel, kernel_size=3, scale=2, conv=default_conv):
        super(RecurrentProjection, self).__init__()
        self.scale = scale
        stride_conv_ksize, stride, padding = {
            2: (6, 2, 2),
            3: (9, 3, 3),
            4: (6, 2, 2)
        }[scale]

        self.multi_source_projection = MultiSourceProjection(in_channel, kernel_size=kernel_size, scale=scale,
                                                             conv=conv)
        self.down_sample_1 = nn.SequentialCell(
            *[nn.Conv2d(in_channel, in_channel, stride_conv_ksize, stride=stride, padding=padding, pad_mode='pad'),
              nn.PReLU(in_channel)]
        )
        if scale != 4:
            self.down_sample_2 = nn.SequentialCell(
                *[nn.Conv2d(in_channel, in_channel, stride_conv_ksize, stride=stride, padding=padding, pad_mode='pad'),
                  nn.PReLU(in_channel)]
            )
        self.error_encode = nn.SequentialCell(
            *[nn.Conv2dTranspose(in_channel, in_channel, stride_conv_ksize, stride=stride, padding=padding,
                                 pad_mode='pad'), nn.PReLU()]
        )
        self.post_conv = BasicBlock(conv, in_channel, in_channel, kernel_size, stride=1, bias=True,
                                    act=nn.PReLU(in_channel))
        if scale == 4:
            self.multi_source_projection_2 = MultiSourceProjection(in_channel, kernel_size=kernel_size, scale=scale,
                                                                   conv=conv)
            self.down_sample_3 = nn.SequentialCell(
                *[nn.Conv2d(in_channel, in_channel, 8, stride=4, padding=2, pad_mode='pad'), nn.PReLU(in_channel)]
            )
            self.down_sample_4 = nn.SequentialCell(
                *[nn.Conv2d(in_channel, in_channel, 8, stride=4, padding=2, pad_mode='pad'), nn.PReLU(in_channel)]
            )
            self.error_encode_2 = nn.SequentialCell(
                *[nn.Conv2dTranspose(in_channel, in_channel, 8, stride=4, padding=2, pad_mode='pad'),
                  nn.PReLU(in_channel)]
            )

    def construct(self, x):
        x_up = self.multi_source_projection(x)
        x_down = self.down_sample_1(x_up)
        error_up = self.error_encode(x - x_down)
        h_estimate = x_up + error_up
        if self.scale == 4:
            x_up_2 = self.multi_source_projection_2(h_estimate)
            x_down_2 = self.down_sample_3(x_up_2)
            error_up_2 = self.error_encode_2(x - x_down_2)
            h_estimate = x_up_2 + error_up_2
            x_final = self.post_conv(self.down_sample_4(h_estimate))
        else:
            x_final = self.post_conv(self.down_sample_2(h_estimate))
        return x_final, h_estimate


class CSNLN(nn.Cell):
    def __init__(self, args, conv=default_conv):
        super(CSNLN, self).__init__()

        n_feats = args.n_feats
        self.depth = args.depth
        kernel_size = 3
        scale = args.scale[0]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        m_head = [
            BasicBlock(conv, args.n_colors, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU(n_feats)),
            BasicBlock(conv, n_feats, n_feats, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU(n_feats))
        ]

        self.SEM = RecurrentProjection(n_feats, scale=scale)

        m_tail = [
            nn.Conv2d(n_feats * self.depth, args.n_colors, kernel_size, padding=(kernel_size // 2), pad_mode='pad')
        ]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.SequentialCell(*m_head)
        self.tail = nn.SequentialCell(m_tail)

    def construct(self, inputs):
        x = self.sub_mean(inputs)
        x = self.head(x)
        bag = []
        for i in range(self.depth):
            x, h_estimate = self.SEM(x)
            bag.append(h_estimate)
        cat = ops.Concat(1)
        h_feature = cat(bag)
        h_final = self.tail(h_feature)
        return self.add_mean(h_final)


if __name__ == "__main__":
    c = CSNLN(config)
