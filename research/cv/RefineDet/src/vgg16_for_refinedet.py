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
"""VGG16 backbone for RefineDet"""

import mindspore.nn  as nn

def _make_conv_layer(channels, use_bn=False, kernel_size=3, stride=1, padding=0):
    """make convolution layers for vgg16"""
    in_channels = channels[0]
    layers = []
    for out_channels in channels[1:]:
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, pad_mode="pad", padding=padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.SequentialCell(layers)

class VGG16_for_RefineDet(nn.Cell):
    """
    VGG-16 network body, reference to caffe model_libs
    """
    def __init__(self):
        super(VGG16_for_RefineDet, self).__init__()
        self.b1 = _make_conv_layer([3, 64, 64], padding=1)
        self.b2 = _make_conv_layer([64, 128, 128], padding=1)
        self.b3 = _make_conv_layer([128, 256, 256, 256], padding=1)
        self.b4 = _make_conv_layer([256, 512, 512, 512], padding=1)
        self.b5 = _make_conv_layer([512, 512, 512, 512], padding=1)
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.m3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.m4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.m5 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.fc6 = nn.Conv2d(in_channels=512, out_channels=1024, pad_mode="pad", padding=3, kernel_size=3, dilation=3)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.relu7 = nn.ReLU()
        self.b6_1 = _make_conv_layer([1024, 256], kernel_size=1)
        self.b6_2 = _make_conv_layer([256, 512], stride=2, padding=1)

    def construct(self, x):
        """construct network"""
        outputs = ()
        x = self.b1(x)
        x = self.m1(x)
        x = self.b2(x)
        x = self.m2(x)
        x = self.b3(x)
        x = self.m3(x)
        x = self.b4(x)
        outputs += (x,)
        x = self.m4(x)
        x = self.b5(x)
        outputs += (x,)
        x = self.m5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        outputs += (x,)
        x = self.relu7(x)
        x = self.b6_1(x)
        x = self.b6_2(x)
        return outputs + (x,)

def vgg16():
    return VGG16_for_RefineDet()
