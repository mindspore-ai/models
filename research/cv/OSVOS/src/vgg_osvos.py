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
"""OSVOS network."""
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net


class CenterCrop(nn.Cell):
    """
    crop the feature maps.
    """
    def __init__(self, height, width):
        super(CenterCrop, self).__init__()
        self.height = height
        self.width = width
        self.stridedSlice = P.StridedSlice()

    def construct(self, x):
        starth = x.shape[2] // 2 - self.height // 2
        startw = x.shape[3] // 2 - self.width // 2
        crop_result = self.stridedSlice(x, (0, 0, starth, startw), (
            x.shape[0], x.shape[1], starth + self.height, startw + self.width), (1, 1, 1, 1))
        return crop_result


class OSVOS(nn.Cell):
    """
    the network of OSVOS.
    """
    def __init__(self, vgg_features_ckpt=None, img_h=480, img_w=854):
        super(OSVOS, self).__init__()

        vgg16_features = make_vgg16_features_layer_mindspore()
        if vgg_features_ckpt is not None:
            param_dict = load_checkpoint(vgg_features_ckpt)
            load_param_into_net(vgg16_features, param_dict)
        vgg16_features_idx = [[0, 4],
                              [4, 9],
                              [9, 16],
                              [16, 23],
                              [23, 30]]
        lay_list = [[64, 64],
                    ['M', 128, 128],
                    ['M', 256, 256, 256],
                    ['M', 512, 512, 512],
                    ['M', 512, 512, 512]]
        print("Constructing OSVOS architecture..")
        stages_ms = nn.CellList()
        side_prep_ms = nn.CellList()
        score_dsn_ms = nn.CellList()
        ms_upscale = nn.CellList()
        ms_upscale_ = nn.CellList()

        for i in range(0, len(lay_list)):
            stages_ms.append(
                vgg16_features[vgg16_features_idx[i][0]:vgg16_features_idx[i][1]])

            if i > 0:
                side_prep_ms.append(nn.Conv2d(
                    lay_list[i][-1], 16, kernel_size=3, padding=1, has_bias=True, pad_mode='pad'))
                score_dsn_ms.append(nn.Conv2d(
                    16, 1, kernel_size=1, padding=0, has_bias=True, pad_mode='pad'))
                ms_upscale_.append(nn.Conv2dTranspose(
                    1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, has_bias=False, pad_mode='pad'))
                ms_upscale.append(nn.Conv2dTranspose(
                    16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, has_bias=False, pad_mode='pad'))

        self.ms_upscale = ms_upscale
        self.ms_upscale_ = ms_upscale_
        self.stages_ms = stages_ms
        self.side_prep_ms = side_prep_ms
        self.score_dsn_ms = score_dsn_ms

        self.center_crop = CenterCrop(img_h, img_w)
        self.concat = P.Concat(axis=1)
        self.fuse = nn.Conv2d(64, 1, kernel_size=1,
                              padding=0, has_bias=True, pad_mode='pad')

    def construct(self, x):
        x = self.stages_ms[0](x)
        side = []
        side_out = []
        for i in range(1, len(self.stages_ms)):
            x = self.stages_ms[i](x)
            side_temp = self.side_prep_ms[i - 1](x)
            side.append(self.center_crop(self.ms_upscale[i - 1](side_temp)))
            side_out.append(self.center_crop(
                self.ms_upscale_[i - 1](self.score_dsn_ms[i - 1](side_temp))))

        out = self.concat(side)
        out = self.fuse(out)
        side_out.append(out)
        return side_out


def make_vgg16_features_layer_mindspore():
    """
    the backbone of vgg net.
    """
    base = [64, 64, 'M', 128, 128, 'M', 256, 256,
            256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=1,
                               pad_mode='pad',
                               has_bias=False,
                               weight_init=weight)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)
