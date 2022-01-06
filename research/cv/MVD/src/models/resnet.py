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
"""ResNet."""

import mindspore.ops as P
import mindspore.common.initializer as init
from mindspore import nn
from mindspore.common.initializer import Normal, Zero
from mindspore.train.serialization import load_checkpoint, load_param_into_net



def weights_init_classifier(module_):
    """
    weight initialization
    """
    classname = module_.__class__.__name__
    if classname.find('Linear') != -1:
        module_.gamma.set_data(
            init(Normal(sigma=0.001), module_.gamma.shape, module_.gamma.dtype))
        if module_.bias:
            module_.bias.set_data(
                init(Zero(), module_.bias.shape, module_.bias.dtype))


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same')


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same')


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same')


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.9, gamma_init=1,
                          beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.997,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.
    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = \
                nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])
        self.add = P.Add()

    def construct(self, x):
        """
        Construct Residual Block
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.
    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_class=395):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.avgpool = P.ReduceMean(keep_dims=True)
        self.bottleneck = nn.BatchNorm1d(num_features=out_channels[3])
        self.classifier = nn.Dense(out_channels[3], num_class)
        weights_init_classifier(self.classifier)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.
        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.
        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        Construct ResNet
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1 = self.maxpool(x)

        conv2 = self.layer1(conv1)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        conv5 = self.layer4(conv4)

        x_pool = self.avgpool(conv5, (2, 3))
        x_pool = x_pool.view(x_pool.shape[0], x_pool.shape[1])
        if self.training:
            feat = self.bottleneck(x_pool)
            logits = self.classifier(feat)
            return feat, logits
        feat = self.bottleneck(x_pool)
        return feat


def resnet50(num_class=395, pretrain=""):
    """
    Get ResNet50 neural network.
    Returns:
        Cell, cell instance of ResNet50 neural network.
    Examples:
        >>> net = resnet50()
    """
    resnet = ResNet(ResidualBlock,
                    [3, 4, 6, 3],
                    [64, 256, 512, 1024],
                    [256, 512, 1024, 2048],
                    [1, 2, 2, 2],
                    num_class=num_class)

    if pretrain:
        param_dict = load_checkpoint(pretrain)
        load_param_into_net(resnet, param_dict)

    return resnet
