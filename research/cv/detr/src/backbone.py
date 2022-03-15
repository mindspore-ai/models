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
"""backbone"""
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import nn
from mindspore import ops


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, pad_mode='pad', group=groups, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


def _bn(channel):
    """bn"""
    bn = nn.BatchNorm2d(channel, eps=1e-5, momentum=0.997, gamma_init=1,
                        beta_init=0, moving_mean_init=0, moving_var_init=1,
                        use_batch_statistics=False)
    _freeze_params(bn)
    return bn


def _bn_last(channel):
    """bn last"""
    bn = nn.BatchNorm2d(channel, eps=1e-5, momentum=0.997, gamma_init=0,
                        beta_init=0, moving_mean_init=0, moving_var_init=1,
                        use_batch_statistics=False)
    _freeze_params(bn)
    return bn


def _freeze_params(layer):
    """freeze params"""
    for par in layer.trainable_params():
        par.requires_grad = False


class Bottleneck(nn.Cell):
    """Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, down_sample_layer=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = _bn(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = _bn(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = _bn_last(planes * self.expansion)
        self.relu = ops.ReLU()
        self.down_sample_layer = down_sample_layer
        self.stride = stride

    def construct(self, x):
        """construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample_layer is not None:
            identity = self.down_sample_layer(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """ResNet"""
    def __init__(self, block, layers, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.num_channels = 2048
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = _bn(self.inplanes)
        self.relu = ops.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        _freeze_params(self.conv1)
        _freeze_params(self.layer1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """make layer"""
        down_sample_layer = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample_layer = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                _bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, down_sample_layer, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Backbone(nn.Cell):
    """backbone with position embedding"""
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels

    def construct(self, images, masks):
        """construct"""
        features = self.backbone(images)
        masks_interpolated = ops.ResizeNearestNeighbor(size=features.shape[-2:])(
            masks[None].astype('float32')
        )[0]
        pos_emb = self.position_embedding(masks_interpolated)
        return features, masks_interpolated, pos_emb


def resnet50(ckpt_path=None):
    """Get ResNet50 neural network.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50()
    """
    net = ResNet(Bottleneck, [3, 4, 6, 3])
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path)
        load_param_into_net(net, ckpt)
    return net
