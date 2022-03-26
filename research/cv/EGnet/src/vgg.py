# Copyright 2021 Huawei Technologies Co., Ltd
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

"""VGG model define"""

import mindspore.nn as nn
from mindspore.train import load_checkpoint
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


class Vgg16(nn.Cell):
    """
    VGG network definition.
    """

    def __init__(self):
        super(Vgg16, self).__init__()
        self.cfg = {"tun": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
                    "tun_ex": [512, 512, 512]}
        self.extract = [8, 15, 22, 29]
        self.extract_ex = [5]
        self.base = nn.CellList(vgg(self.cfg["tun"], 3))
        self.base_ex = VggEx(self.cfg["tun_ex"], 512)

    def load_pretrained_model(self, model_file):
        load_checkpoint(model_file, net=self.base)

    def construct(self, x, multi=0):
        """construct"""
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)
        x = self.base_ex(x)
        tmp_x.append(x)
        if multi == 1:
            tmp_y = [tmp_x[0]]
            return tmp_y
        return tmp_x


class VggEx(nn.Cell):
    """ VGGEx block. """

    def __init__(self, cfg, incs=512, padding=1, dilation=1):
        super(VggEx, self).__init__()
        self.cfg = cfg
        layers = []
        for v in self.cfg:
            conv2d = nn.Conv2d(incs, v, kernel_size=3, pad_mode="pad", padding=padding, dilation=dilation,
                               has_bias=False)
            layers += [conv2d, nn.ReLU()]
            incs = v
        self.ex = nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.ex(x)
        return x


if __name__ == "__main__":
    # print VGG network
    net = Vgg16()
    num_params = 0
    for m in net.get_parameters():
        print(m)
        num_params += m.size
    print(net)
    print("The number of parameters: {}".format(num_params))
