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


"""arbitrary style transfer network."""
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import Parameter
from mindspore import ops
from mindspore.common import initializer as init


#########################################################################
#                               module
#########################################################################
class InstanceNorm2d(nn.Cell):
    """myown InstanceNorm2d"""

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.moving_mean = Parameter(init.initializer('zeros', num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(init.initializer('ones', num_features), name="variance", requires_grad=False)
        self.gamma = Parameter(init.initializer(gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(init.initializer(beta_init, num_features), name="beta", requires_grad=affine)

    def construct(self, x):
        """calculate InstanceNorm output"""
        mean = ops.ReduceMean(keep_dims=True)(x, (2, 3))
        var = ops.ReduceMean(keep_dims=True)(((x - mean) ** 2), (2, 3))
        std = (var + self.eps) ** 0.5
        x = (x - mean) / std * self.gamma.reshape(1, -1, 1, 1) + self.beta.reshape(1, -1, 1, 1)
        return x


class style_prediction_network(nn.Cell):
    """
    style_prediction_network.

    Args:
        style_dim(int): The dimension of style vector. Default: 100.
    Returns:
        Tensor, output tensor.
    """

    def __init__(self, args, style_dim=100):
        super(style_prediction_network, self).__init__()
        if args.platform == 'GPU':
            self.fc = nn.Dense(768, style_dim)
        elif args.platform == 'Ascend':
            self.fc = nn.Dense(768, style_dim).to_float(mstype.float16)
        self.flatten = nn.Flatten()

    def construct(self, style_feat):
        """construct"""
        style_vector = style_feat.mean(axis=3).mean(axis=2)
        style_vector = self.flatten(style_vector)
        style_vector = self.fc(style_vector)
        return style_vector


class style_transfer_network(nn.Cell):
    """
    style_prediction_network.

    Args:
        style_dim(int): The dimension of style vector. Default: 100.
    Returns:
            Tensor, the stylied image.
    """

    def __init__(self, args, style_dim=100):
        super(style_transfer_network, self).__init__()
        self.conv1 = ConvInRelu(3, 32, 9, stride=1)
        self.conv2 = ConvInRelu(32, 64, 3, stride=2)
        self.conv3 = ConvInRelu(64, 128, 3, stride=2)
        self.res1 = ResidualBlockWithStyle(args, 128, style_dim=style_dim)
        self.res2 = ResidualBlockWithStyle(args, 128, style_dim=style_dim)
        self.res3 = ResidualBlockWithStyle(args, 128, style_dim=style_dim)
        self.res4 = ResidualBlockWithStyle(args, 128, style_dim=style_dim)
        self.res5 = ResidualBlockWithStyle(args, 128, style_dim=style_dim)
        self.upconv1 = UpsampleConvInReluWithStyle(args, 128, 64, 3, stride=1, upsample=2, style_dim=style_dim)
        self.upconv2 = UpsampleConvInReluWithStyle(args, 64, 32, 3, stride=1, upsample=2, style_dim=style_dim)
        self.upconv3 = UpsampleConvInReluWithStyle(args, 32, 3, 9, stride=1, upsample=None, style_dim=style_dim,
                                                   activation=None)

        self.tanh = nn.Tanh()  # The original paper used the sigmoid function

        if args.platform == 'Ascend':
            self.res1 = self.res1.to_float(mstype.float16)
            self.res2 = self.res2.to_float(mstype.float16)
            self.res3 = self.res3.to_float(mstype.float16)
            self.res4 = self.res4.to_float(mstype.float16)
            self.res5 = self.res5.to_float(mstype.float16)
            self.upconv1 = self.upconv1.to_float(mstype.float16)
            self.upconv2 = self.upconv2.to_float(mstype.float16)
            self.upconv3 = self.upconv3.to_float(mstype.float16)

    def construct(self, content, style_vector):
        """construct"""
        x = self.conv1(content)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x, style_vector)
        x = self.res2(x, style_vector)
        x = self.res3(x, style_vector)
        x = self.res4(x, style_vector)
        x = self.res5(x, style_vector)
        x = self.upconv1(x, style_vector)
        x = self.upconv2(x, style_vector)
        x = self.upconv3(x, style_vector)
        x = self.tanh(x)
        return x


#########################################################################
#                            Basic block
#########################################################################
class ConvInRelu(nn.Cell):
    """
        Convolution fused with InstanceNorm and ReLU block definition.
        Args:
            channels_in (int): Input channel.
            channels_out (int): Output channel.
            kernel_size (int): Input kernel size.
            stride (int): Stride size for the first convolutional layer. Default: 1.
        Returns:
            Tensor, output tensor.
        """

    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        super(ConvInRelu, self).__init__()
        pad_mode = "REFLECT"
        pad_mode = "CONSTANT"
        padding = (kernel_size - 1) // 2
        paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        self.pad = nn.Pad(paddings=paddings, mode=pad_mode)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, pad_mode='pad')
        self.instancenorm = InstanceNorm2d(channels_out)
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        x = self.pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x


class ResidualBlockWithStyle(nn.Cell):
    """
        ResidualBlockWithStyle fused with ResidualBlock and style_vector.
        Args:
            channels (int): Input channel and output channel.
            style_dim (int): The dimension of style vector. Default: 100.
        Returns:
            Tensor, output tensor.
        """

    def __init__(self, args, channels, style_dim=100):
        super(ResidualBlockWithStyle, self).__init__()
        self.channels = channels

        self.fc_beta1 = nn.Dense(style_dim, channels)
        self.fc_gamma1 = nn.Dense(style_dim, channels)
        self.fc_beta2 = nn.Dense(style_dim, channels)
        self.fc_gamma2 = nn.Dense(style_dim, channels)

        pad_mode = "REFLECT"
        pad_mode = "CONSTANT"
        padding = 1
        paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        self.pad = nn.Pad(paddings=paddings, mode=pad_mode)

        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, pad_mode='pad')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, pad_mode='pad')
        if args.platform == 'GPU':
            self.instancenorm1 = nn.InstanceNorm2d(channels)
            self.instancenorm2 = nn.InstanceNorm2d(channels)
        elif args.platform == 'Ascend':
            self.instancenorm1 = InstanceNorm2d(channels)
            self.instancenorm2 = InstanceNorm2d(channels)

    def construct(self, x, style):
        """construct"""
        # x: B x C x H x W
        # style: B x style_dim
        expand_dims = ops.ExpandDims()
        beta1 = self.fc_beta1(style)
        beta1 = expand_dims(beta1, 2)  # beta1: B x C x 1
        beta1 = expand_dims(beta1, 3)  # beta1: B x C x 1 x 1

        gamma1 = self.fc_gamma1(style)
        gamma1 = expand_dims(gamma1, 2)
        gamma1 = expand_dims(gamma1, 3)

        beta2 = self.fc_beta2(style)
        beta2 = expand_dims(beta2, 2)
        beta2 = expand_dims(beta2, 3)

        gamma2 = self.fc_gamma2(style)
        gamma2 = expand_dims(gamma2, 2)
        gamma2 = expand_dims(gamma2, 3)

        y = self.pad(x)
        y = self.conv1(y)
        y = self.instancenorm1(y)
        y = gamma1 * y
        y += beta1
        y = self.relu1(y)
        y = self.pad(y)
        y = self.conv2(y)
        y = self.instancenorm2(y)
        y = gamma2 * y
        y += beta2
        return x + y


class UpsampleConvInReluWithStyle(nn.Cell):
    """
        UpsampleConvInReluWithStyle fused with UpsampleConvInRelu and style_vector.
        Args:
            channels_in (int): Input channel.
            channels_out (int): Output channel.
            kernel_size (int): Input kernel size.
            stride (int): Stride size for the first convolutional layer. Default: 1.
            upsample (int):The scale of the upsampling. Default: 2.
            style_dim (int): The dimension of style vector.Default: 100.
            activation(nn.Cell) : The specific activation function used. Default: nn.ReLU.
        Returns:
            Tensor, output tensor.
        """

    def __init__(self, args, channels_in, channels_out, kernel_size, stride=1, upsample=2, style_dim=100,
                 activation=nn.ReLU):
        super(UpsampleConvInReluWithStyle, self).__init__()
        self.upsample = upsample
        if self.upsample:
            if args.platform == 'GPU':
                self.upsample_layer = nn.ResizeBilinear()
            elif args.platform == 'Ascend':
                self.upsample_layer = nn.ResizeBilinear().to_float(mstype.float16)

        pad_mode = "CONSTANT"  # The original paper uses reflect, but the performance of reflect in mindspore r1.3 is too poor
        padding = (kernel_size - 1) // 2
        paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        self.pad = nn.Pad(paddings=paddings, mode=pad_mode)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, pad_mode='pad')
        if args.platform == 'GPU':
            self.instancenorm = nn.InstanceNorm2d(channels_out)
        elif args.platform == 'Ascend':
            self.instancenorm = InstanceNorm2d(channels_out)
        self.fc_beta = nn.Dense(style_dim, channels_out)
        self.fc_gamma = nn.Dense(style_dim, channels_out)
        if activation:
            self.activation = activation()
        else:
            self.activation = None

    def construct(self, x, style):
        """construct"""
        # x: B x C_in x H x W
        # style: B x style_dim
        expand_dims = ops.ExpandDims()
        beta = self.fc_beta(style)
        beta = expand_dims(beta, 2)  # beta: B x C x 1
        beta = expand_dims(beta, 3)  # beta: B x C x 1 x 1

        gamma = self.fc_gamma(style)
        gamma = expand_dims(gamma, 2)
        gamma = expand_dims(gamma, 3)

        if self.upsample:
            x = self.upsample_layer(x, scale_factor=self.upsample)
        x = self.pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = gamma * x
        x += beta
        if self.activation:
            x = self.activation(x)
        return x


#########################################################################
#                            init_weights
#########################################################################
def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, (nn.BatchNorm2d, InstanceNorm2d)):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
