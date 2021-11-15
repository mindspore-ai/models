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
'''
Alphapose network
'''
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from src.SE_module import SELayer

class MPReverse(nn.Cell):
    '''
    MPReverse
    '''
    def __init__(self, kernel_size=1, stride=1, pad_mode="valid"):
        super(MPReverse, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.reverse = ops.ReverseV2(axis=[2, 3])

    def construct(self, x):
        x = self.reverse(x)
        x = self.maxpool(x)
        x = self.reverse(x)
        return x

class Bottleneck(nn.Cell):
    '''
    model part of network
    '''
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               pad_mode='pad',padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.1)
        if reduction:
            self.se = SELayer(planes * 4)
            
        self.relu = nn.ReLU()
        self.reduc = reduction
        self.down_sample_layer = downsample
        self.stride = stride

    def construct(self, x):
        '''
        construct
        '''
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)
        if self.down_sample_layer is not None:
            residual = self.down_sample_layer(x)
        out = out+residual
        out = self.relu(out)
        return out

class SEResnet(nn.Cell):
    '''
    model part of network
    '''
    def __init__(self, architecture):
        super(SEResnet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, pad_mode='pad',padding=3, has_bias=False)
        
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU()
        self.maxpool = MPReverse(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2)

    def construct(self, x): 
        '''
        construct
        '''
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x) 
        return x
    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]
        
    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([nn.Conv2d(self.inplanes, planes * block.expansion,
                                                      kernel_size=1, stride=stride, has_bias=False),
                                            nn.BatchNorm2d(planes * block.expansion, momentum=0.1)])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(layers)

