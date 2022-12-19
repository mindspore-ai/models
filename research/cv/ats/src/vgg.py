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

"""cifar vgg."""


import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops


def get_vgg_cfg(vgg_n_layer):
    if vgg_n_layer == 8:
        cfg = [[64], [128], [256], [512], [512]]
    elif vgg_n_layer == 11:
        cfg = [[64], [128], [256, 256], [512, 512], [512, 512]]
    elif vgg_n_layer == 13:
        cfg = [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]]
    elif vgg_n_layer == 16:
        cfg = [
            [64, 64], [128, 128], [256, 256, 256],
            [512, 512, 512], [512, 512, 512]
        ]
    elif vgg_n_layer == 19:
        cfg = [
            [64, 64], [128, 128], [256, 256, 256, 256],
            [512, 512, 512, 512], [512, 512, 512, 512]
        ]
    else:
        raise ValueError("No such vgg_n_layer: {}".format(vgg_n_layer))

    return cfg


class CifarVGG(nn.Cell):

    def __init__(self, vgg_n_layer, n_classes, use_bn=True):
        super(CifarVGG, self).__init__()
        cfg = get_vgg_cfg(vgg_n_layer)

        self.block0 = self._make_layers(cfg[0], use_bn, 3)
        self.block1 = self._make_layers(cfg[1], use_bn, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], use_bn, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], use_bn, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], use_bn, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AvgPool2d(4)

        self.relu = ops.ReLU()
        self.classifier = nn.Dense(512, n_classes)

    def construct(self, x0):
        f = self.relu(self.block0(x0))
        f = self.pool0(f)

        f = self.relu(self.block1(f))
        f = self.pool1(f)

        f = self.relu(self.block2(f))
        f = self.pool2(f)

        f = self.relu(self.block3(f))
        f = self.relu(self.block4(f))

        f = self.avgpool(f)

        f = f.view(f.shape[0], -1)
        g = self.classifier(f)

        return g

    def _make_layers(self, cfg, use_bn=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1, pad_mode="pad"
                )
                if use_bn:
                    layers += [
                        conv2d, nn.BatchNorm2d(v), nn.ReLU()
                    ]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        layers = layers[:-1]
        print("Make layers done...", self.relu)
        return nn.SequentialCell(layers)

    def forward(self):
        print("Not implemented...", self.relu)
        return self.relu


if __name__ == '__main__':
    for size in [32]:
        for n_layer in [8, 11, 13, 16, 19]:
            x = Tensor(np.random.randn(2, 3, size, size), mindspore.float32)
            model = CifarVGG(vgg_n_layer=n_layer, n_classes=100)
            feats, logits = model.construct(x)

            n_params = sum([
                np.prod(param.shape) for param in model.get_parameters()
            ])
            print("Total number of parameters : {}".format(
                n_params
            ))

            print(feats.shape, logits.shape)
