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


import torch
import torch.nn as nn


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


def vgg16():
    cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
           'tun_ex': [512, 512, 512]}
    return torch.nn.ModuleList(vgg(cfg['tun'], 3))
