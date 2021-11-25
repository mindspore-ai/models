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
"""Backbone"""
import mindspore.nn as nn


class Bottleneck(nn.Cell):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1,
                 use_dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False, dilation=dilation, pad_mode='pad')
        self.bn1 = norm_layer(planes, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        if use_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad',
                                   padding=dilation, has_bias=False, dilation=dilation)

        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad',
                                   padding=dilation, has_bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False, dilation=dilation, pad_mode='pad')
        self.bn3 = norm_layer(planes * 4, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """Forward"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Cell):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, dcn_layers=None, dcn_interval=1, atrous_layers=None, block=Bottleneck,
                 norm_layer=nn.BatchNorm2d):
        """
        dcn_layers have and must have four elements, by default, dcn_layers will be [0, 0, 0, 0]
        """
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.CellList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers
        self.dcn_layers = dcn_layers if not dcn_layers is None else [0, 0, 0, 0]

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3, 3, 3), pad_mode='pad', has_bias=False)
        self.bn1 = self.norm_layer(64, eps=1e-05, momentum=0.9, affine=True, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self._make_layer(block, 64, layers[0], dcn_layers=self.dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 128, layers[1], stride=2, dcn_layers=self.dcn_layers[1], dcn_interval=dcn_interval)
        self._make_layer(block, 256, layers[2], stride=2, dcn_layers=self.dcn_layers[2], dcn_interval=dcn_interval)
        self._make_layer(block, 512, layers[3], stride=2, dcn_layers=self.dcn_layers[3], dcn_interval=dcn_interval)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_cells = [m for m in self.cells() if isinstance(m, nn.Conv2d)]
        # self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    # Generate a stage
    def _make_layer(self, block, planes, blocks, stride=1, dcn_layers=0, dcn_interval=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.atrous_layers and len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1

            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False,
                          dilation=self.dilation, pad_mode='pad'),
                self.norm_layer(planes * block.expansion, eps=1e-05, momentum=0.9,
                                affine=True, use_batch_statistics=True),
            )

        layers = []
        use_dcn = (dcn_layers >= blocks)
        # Generate a block of stage
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation, use_dcn=use_dcn))
        self.inplanes = planes * block.expansion
        # Generate other blocks of the stage
        for i in range(1, blocks):
            use_dcn = ((i + dcn_layers) >= blocks) and (i % dcn_interval == 0)
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, use_dcn=use_dcn))
        layer = nn.SequentialCell(*layers)
        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def construct(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = ()
        for layer in self.layers:
            x = layer(x)
            outs += (x,)
        return outs

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)



def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """

    backbone = cfg['type'](*cfg['args'])

    num_layers = max(cfg['selected_layers']) + 1


    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone
