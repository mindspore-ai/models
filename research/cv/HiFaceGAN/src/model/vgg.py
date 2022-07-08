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
"""VGG network"""
import mindspore.nn as nn

_VGG19_LAYERS = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M')


def _make_layer(base, args, batch_norm):
    """Make stage network of VGG"""
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
                               padding=args['padding'],
                               pad_mode=args['pad_mode'],
                               has_bias=args['has_bias'],
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, use_batch_statistics=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.CellList(layers)


class Vgg19(nn.Cell):
    """VGG 19-layer model with Batch normalization"""

    def __init__(self):
        super().__init__()
        self.args = {
            'padding': 0,
            'pad_mode': 'same',
            'has_bias': False
        }
        self.layers = _make_layer(_VGG19_LAYERS, self.args, batch_norm=True)

    def construct(self, x):
        """Feed forward"""
        for cell in self.layers:
            x = cell(x)
        return x
