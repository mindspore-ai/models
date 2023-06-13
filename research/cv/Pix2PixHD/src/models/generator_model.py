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
# ===========================================================================

"""
    Define generator  Generator.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from src.utils.config import config


class ResnetBlock(nn.Cell):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        padding = 0
        if padding_type in ("REFLECT", "SYMMETRIC", "CONSTANT"):
            paddings = ((0, 0), (0, 0), (1, 1), (1, 1))
            conv_block += [nn.Pad(paddings=paddings, mode=padding_type)]
        else:
            padding = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=padding, pad_mode="valid", has_bias=True),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(p=0.5)]

        if padding_type in ("REFLECT", "SYMMETRIC", "CONSTANT"):
            paddings = ((0, 0), (0, 0), (1, 1), (1, 1))
            conv_block += [nn.Pad(paddings=paddings, mode=padding_type)]

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=padding, pad_mode="valid", has_bias=True),
            norm_layer(dim),
        ]

        return nn.SequentialCell(conv_block)

    def construct(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Cell):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.BatchNorm2d,
        padding_type="CONSTANT",
    ):
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU()
        paddings = ((0, 0), (0, 0), (3, 3), (3, 3))

        model = [
            nn.Pad(paddings=paddings, mode=padding_type),
            nn.Conv2d(input_nc, ngf, kernel_size=7, pad_mode="valid", has_bias=True),
            norm_layer(ngf),
            activation,
        ]
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        # resnet blocks
        mult = 2**n_downsampling
        while n_blocks > 0:
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            n_blocks -= 1

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Conv2dTranspose(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, has_bias=True),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]

        model += [
            nn.Pad(paddings=paddings, mode=padding_type),
            nn.Conv2d(ngf, output_nc, kernel_size=7, pad_mode="valid", has_bias=True),
            nn.Tanh(),
        ]
        self.model = nn.SequentialCell(model)

    def construct(self, input_data):
        return self.model(input_data)


class LocalEnhancer(nn.Cell):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=32,
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3,
        norm_layer=nn.BatchNorm2d,
        padding_type="CONSTANT",
    ):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        # global generator model
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(
            input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer
        ).model
        # get rid of final convolution layers
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = nn.SequentialCell(model_global)

        # local enhancer layers
        for n in range(1, n_local_enhancers + 1):
            # downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            paddings = ((0, 0), (0, 0), (3, 3), (3, 3))
            model_downsample = [
                nn.Pad(paddings=paddings, mode=padding_type),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, pad_mode="valid", has_bias=True),
                norm_layer(ngf_global),
                nn.ReLU(),
                nn.Conv2d(
                    ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True
                ),
                norm_layer(ngf_global * 2),
                nn.ReLU(),
            ]
            # residual blocks
            model_upsample = []
            num = n_blocks_local
            while num > 0:
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
                num -= 1
            # upsample
            model_upsample += [
                nn.Conv2dTranspose(ngf_global * 2, ngf_global, kernel_size=3, stride=2, has_bias=True),
                norm_layer(ngf_global),
                nn.ReLU(),
            ]

            # final convolution
            if n == n_local_enhancers:
                model_upsample += [
                    nn.Pad(paddings=paddings, mode=padding_type),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, pad_mode="valid", has_bias=True),
                    nn.Tanh(),
                ]

            self.model_local_downsample = nn.SequentialCell(model_downsample)
            self.model_local_upsample = nn.SequentialCell(model_upsample)

        self.downsample = nn.AvgPool2d(3, stride=2, pad_mode="same")

    def construct(self, input_data):
        # create input pyramid
        input_downsampled = [input_data]
        num = self.n_local_enhancers
        while num > 0:
            input_downsampled.append(self.downsample(input_downsampled[-1]))
            num -= 1
        # output at coarest level
        output_prev = self.model(input_downsampled[-1])
        # build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = self.model_local_downsample
            model_upsample = self.model_local_upsample
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class Encoder(nn.Cell):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc
        self.logical_not = ops.LogicalNot()
        self.reduce_sum = ops.ReduceSum()
        self.div = ops.Div()
        self.shape = ops.TensorShape()

        paddings = ((0, 0), (0, 0), (3, 3), (3, 3))

        model = [
            nn.Pad(paddings=paddings, mode="CONSTANT"),
            nn.Conv2d(input_nc, ngf, kernel_size=7, pad_mode="valid", has_bias=True),
            norm_layer(ngf),
            nn.ReLU(),
        ]
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(),
            ]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Conv2dTranspose(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, has_bias=True),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(),
            ]

        model += [
            nn.Pad(paddings=paddings, mode="CONSTANT"),
            nn.Conv2d(ngf, output_nc, kernel_size=7, pad_mode="valid", has_bias=True),
            nn.Tanh(),
        ]
        self.model = nn.SequentialCell(model)

    def construct(self, input_data, inst):
        output = self.model(input_data)
        # instance-wise average pooling
        outputs = output.copy()
        outputs_b = outputs.copy()
        outputs_j = outputs.copy()

        inst_unique = ops.unique(inst)[0]
        inst_unique_shape = self.shape(inst_unique)
        for b in range(input_data.shape[0]):
            for j in range(self.output_nc):
                index_i = Tensor(0, ms.int32)
                outputs_mean_temp = outputs[b : b + 1, j : j + 1, :, :]
                while index_i < inst_unique_shape[0]:
                    inst_index_mask = inst[b : b + 1] == inst_unique[index_i]
                    inst_index_mask_not = self.logical_not(inst_index_mask)
                    outputs_mean_temp_mul = outputs_mean_temp * inst_index_mask
                    outputs_mean_temp_sum = self.reduce_sum(outputs_mean_temp_mul)
                    inst_index_mask_to_int = inst_index_mask.astype(ms.int32)
                    count_nonzero_num = ops.count_nonzero(inst_index_mask_to_int)
                    if count_nonzero_num > 0:
                        mean = self.div(outputs_mean_temp_sum, count_nonzero_num)
                        outputs_mean_temp = mean * inst_index_mask + outputs_mean_temp * inst_index_mask_not
                    index_i += 1
                if j == 0:
                    outputs_j = outputs_mean_temp
                else:
                    outputs_j = ops.Concat(axis=1)((outputs_j, outputs_mean_temp))
            if b == 0:
                outputs_b = outputs_j
            else:
                outputs_b = ops.Concat(axis=0)((outputs_b, outputs_j))
        return outputs_b


class Vgg19(nn.Cell):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).layers
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        for x in range(2):
            self.slice1.append(vgg_pretrained_features.cell_list[x])
        for x in range(2, 7):
            self.slice2.append(vgg_pretrained_features.cell_list[x])
        for x in range(7, 12):
            self.slice3.append(vgg_pretrained_features.cell_list[x])
        for x in range(12, 21):
            self.slice4.append(vgg_pretrained_features.cell_list[x])
        for x in range(21, 30):
            self.slice5.append(vgg_pretrained_features.cell_list[x])
        if not requires_grad:
            for param in vgg_pretrained_features.get_parameters():
                param.requires_grad = False

    def construct(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def _make_layer(base, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)


class Vgg(nn.Cell):
    """
    VGG network features definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        batch_norm (bool): Whether to do the batchnorm. Default: False.

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        >>>     batch_norm=False)
    """

    def __init__(self, base, batch_norm=False):
        super(Vgg, self).__init__()
        self.layers = _make_layer(base, batch_norm=batch_norm)

    def construct(self, x):
        x = self.layers(x)
        return x


cfg = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg19(pretrained=False):
    """
    Get Vgg19 features neural network with Batch Normalization.

    Args:
        pretrained: Load pre training weight. Default:False

    Returns:
        Cell, cell instance of Vgg19 neural network with Batch Normalization.

    Examples:
        >>> vgg19()
    """
    net = Vgg(cfg["19"])
    if pretrained:
        load_param_into_net(net, load_checkpoint(config.vgg_pre_trained))
    return net
