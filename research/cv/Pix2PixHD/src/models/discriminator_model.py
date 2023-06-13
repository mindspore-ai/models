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
    Define discriminator model——Patch GAN.
"""

import mindspore.nn as nn


class ConvNormReLU(nn.Cell):
    """
    Convolution fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        use_relu (bool): Use relu or not. Default: True.
        padding (int): Pad size, if it is None, it will calculate by kernel_size. Default: None.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=4,
        stride=2,
        alpha=0.2,
        norm_layer=nn.BatchNorm2d,
        use_relu=True,
        padding=None,
    ):
        super(ConvNormReLU, self).__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode="pad", padding=padding, has_bias=True)
        layers = [conv, norm_layer(out_planes)]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class Discriminator(nn.Cell):
    """
    Discriminator of Model.

    Args:
        in_planes (int): Input channel.
        ndf (int): the number of filters in the last conv layer
        n_layers (int): The number of ConvNormReLU blocks.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        use_sigmoid (bool):
        get_inner_feat (bool):
        alpha (float): LeakyRelu slope. Default: 0.2.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self,
        in_planes=3,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        get_inner_feat=False,
        alpha=0.2,
    ):
        super(Discriminator, self).__init__()
        self.get_inner_feat = get_inner_feat
        self.n_layers = n_layers
        kernel_size = 4
        layers = [
            [nn.Conv2d(in_planes, ndf, kernel_size, 2, pad_mode="pad", padding=2, has_bias=True), nn.LeakyReLU(alpha)]
        ]
        nf_mult = ndf
        i = 1
        while i < n_layers:
            nf_mult_prev = nf_mult
            nf_mult = min(nf_mult * 2, 512)
            layers.append([ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 2, alpha, norm_layer, padding=2)])
            i = i + 1

        nf_mult_prev = nf_mult
        nf_mult = min(nf_mult * 2, 512)
        layers.append([ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 1, alpha, norm_layer, padding=2)])
        layers.append([nn.Conv2d(nf_mult, 1, kernel_size, 1, pad_mode="pad", padding=2, has_bias=True)])
        if use_sigmoid:
            layers.append([nn.Sigmoid()])
        if get_inner_feat:
            for n in range(len(layers)):
                setattr(self, "d_inner_features" + str(n), nn.SequentialCell(layers[n]))
        else:
            self.features = nn.SequentialCell([i for layer in layers for i in layer])

    def construct(self, input_data):
        if self.get_inner_feat:
            output = [input_data]
            for n in range(self.n_layers + 2):
                features = getattr(self, "d_inner_features" + str(n))
                output.append(features(output[-1]))
            return output[1:]
        return self.features(input_data)


class MultiscaleDiscriminator(nn.Cell):
    """
    MultiscaleDiscriminator of Model.

    Args:
        in_planes (int): Input channel.
        ndf (int): the number of filters in the last conv layer
        n_layers (int): The number of ConvNormReLU blocks.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        use_sigmoid (bool):
        get_inner_feat (bool):

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self, in_planes, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_d=3, get_inner_feat=False
    ):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_d = num_d
        self.n_layers = n_layers
        self.get_inner_feat = get_inner_feat

        for i in range(num_d):
            net_d = Discriminator(in_planes, ndf, n_layers, norm_layer, use_sigmoid, get_inner_feat)
            if get_inner_feat:
                for j in range(n_layers + 2):
                    setattr(self, "scale" + str(i) + "_layer" + str(j), getattr(net_d, "d_inner_features" + str(j)))
            else:
                setattr(self, "layer" + str(i), net_d.features)

        self.downsample = nn.AvgPool2d(3, stride=2, pad_mode="same")

    def singleD_forward(self, model, input_data):
        if self.get_inner_feat:
            result = [input_data]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        return [model(input_data)]

    def construct(self, input_data):
        result = []
        input_downsampled = input_data
        for i in range(self.num_d):
            if self.get_inner_feat:
                model = []
                for j in range(self.n_layers + 2):
                    model.append(getattr(self, "scale" + str(self.num_d - 1 - i) + "_layer" + str(j)))
            else:
                model = getattr(self, "layer" + str(self.num_d - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))

            if i != (self.num_d - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
