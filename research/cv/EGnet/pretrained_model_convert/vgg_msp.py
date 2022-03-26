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
import mindspore


def vgg(cfg, i, batch_norm=False):
    """Make stage network of VGG."""
    layers = []
    in_channels = i
    stage = 1
    pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1))).to_float(mindspore.dtype.float32)
    for v in cfg:
        if v == "M":
            stage += 1
            layers += [pad, nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode="pad", padding=1, has_bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return layers


# adding prefix "base" to parameter names for load_checkpoint().
class Tmp(nn.Cell):
    def __init__(self, base):
        super(Tmp, self).__init__()
        self.base = base


def vgg16():
    cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
           'tun_ex': [512, 512, 512]}
    base = nn.CellList(vgg(cfg['tun'], 3))
    base = Tmp(base)
    return Tmp(base)
