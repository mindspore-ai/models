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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, XavierUniform

ms.set_seed(0)

class MyBatchNorm1d(nn.Cell):
    def __init__(
            self, num_features, eps=1e-5, momentum=0.9, affine=True, gamma_init="ones",
            beta_init="zeros", moving_mean_init="zeros", moving_var_init="ones", use_batch_statistics=None,
    ):
        super(MyBatchNorm1d, self).__init__()
        self.norm2d = nn.BatchNorm2d(num_features=num_features)
        self.expand_dims = ops.operations.ExpandDims()
        self.squeeze = ops.operations.Squeeze(3)
        self.print = ops.operations.Print()
    def construct(self, x):
        out = x
        if len(x.shape) == 3:
            out = self.expand_dims(out, 3)
            out = self.norm2d(out)
            out = self.squeeze(out)
        return out

class TDNNBlock(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            bias,
            activation=nn.ReLU,
            groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=bias,
            weight_init='he_uniform',
            bias_init='truncatedNormal'
        )
        self.activation = activation()
        self.norm = MyBatchNorm1d(num_features=out_channels)
        self.print = ops.operations.Print()

    def construct(self, x):
        out = self.conv(x)
        out = self.activation(out)
        out = self.norm(out)
        return out

class Res2NetBlock(nn.Cell):
    def __init__(
            self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1, bias=True, groups=1
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.CellList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale
        self.cat = ms.ops.Concat(axis=1)
        self.split = ms.ops.Split(1, scale)
        self.print = ops.operations.Print()
    def construct(self, x):
        y = []
        spx = self.split(x)
        y_i = x
        for i, x_i in enumerate(spx):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = self.cat(y)
        return y

class SEBlock(nn.Cell):
    """An implementation of squeeze-and-excitation block.
    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1, has_bias=True, weight_init='he_uniform',
            bias_init='truncatedNormal'
        )
        self.relu = ms.nn.ReLU()
        self.conv2 = ms.nn.Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1, has_bias=True, weight_init='he_uniform',
            bias_init='truncatedNormal'
        )
        self.sigmoid = ms.nn.Sigmoid()
        self.print = ops.operations.Print()
    def construct(self, x, lengths=None):
        s = x.mean((2), True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x

class SERes2NetBlock(nn.Cell):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.
    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : mindspore class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            res2net_scale=8,
            se_channels=128,
            kernel_size=1,
            dilation=1,
            activation=nn.ReLU,
            bias=True,
            groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            bias=bias,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation, bias
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            bias=bias,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                weight_init='he_uniform',
                bias_init='truncatedNormal'
            )

    def construct(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual

class AttentiveStatisticsPooling(nn.Cell):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    """

    def __init__(self, channels, attention_channels=128, bias=True, global_context=False):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1, bias=bias)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1, bias=bias)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1,
            has_bias=bias, weight_init='he_uniform', bias_init='truncatedNormal'
        )
        self.sqrt = ms.ops.Sqrt()
        self.pow = ms.ops.Pow()
        self.expandDim = ms.ops.ExpandDims()
        self.softmax = ms.ops.Softmax(axis=2)
        self.cat = ms.ops.Concat(axis=1)
        self.print = ops.operations.Print()
        self.ones = ms.ops.Ones()
        self.tile = ms.ops.Tile()
    def construct(self, x, lengths=None):
        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)

            std = self.sqrt(((m * self.pow((x -self.expandDim(mean, dim)), 2)).sum(dim)).clip(eps, None))
            return mean, std
        attn = x
        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = self.softmax(attn)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = self.cat((mean, std))
        pooled_stats = self.expandDim(pooled_stats, 2)

        return pooled_stats

class ECAPA_TDNN(nn.Cell):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : mindspore class
        A class for constructing the activation layers.
    channels : tuple of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : tuple of ints
        tuple of kernel sizes for each layer.
    dilations : tuple of ints
        tuple of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : tuple of ints
        tuple of groups for kernels in each layer.
    """

    def __init__(
            self,
            input_size,
            lin_neurons=192,
            activation=nn.ReLU,
            channels=(512, 512, 512, 512, 1536),
            kernel_sizes=(5, 3, 3, 3, 1),
            dilations=(1, 2, 3, 4, 1),
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=False,
            groups=(1, 1, 1, 1, 1),
    ):

        super().__init__()
        self.input_size = input_size
        self.channels = channels
        self.blocks = nn.CellList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                True,
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    bias=True,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            True,
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            bias=True,
            global_context=global_context,
        )
        self.asp_bn = MyBatchNorm1d(num_features=channels[-1] * 2)

        # Final linear transformation
        self.fc = nn.Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
            has_bias=True,
            weight_init='he_uniform',
            bias_init='truncatedNormal'
        )
        self.expandDim = ms.ops.ExpandDims()
        self.softmax = ms.ops.Softmax(axis=2)
        self.cat = ms.ops.Concat(axis=1)
        self.print = ops.operations.Print()
        self.transpose = ms.ops.Transpose()

    def construct(self, x, lengths=None):
        # Minimize transpose for efficiency
        x = self.transpose(x, (0, 2, 1))
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        layer_tmp = []
        for idx in range(1, len(xl)):
            layer_tmp.append(xl[idx])
        x = self.cat(layer_tmp)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        return x.squeeze()


class Classifier(nn.Cell):
    """This class implements the cosine similarity on the top of features.
    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.
    """

    def __init__(
            self,
            input_size,
            lin_blocks=0,
            lin_neurons=192,
            out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.CellList()

        for _ in range(lin_blocks):
            self.blocks.extend(
                [
                    MyBatchNorm1d(num_features=input_size),
                    nn.Dense(in_channels=input_size, out_channels=lin_neurons),
                ]
            )
            input_size = lin_neurons
        input_size = lin_neurons
        # Final Layer
        tensor1 = initializer(XavierUniform(), [out_neurons, input_size], ms.float32)
        self.weight = ms.Parameter(
            tensor1
        )
        self.norm = ms.ops.L2Normalize(axis=1)
        self.print = ops.operations.Print()
        self.matmul = ms.ops.MatMul()
        self.expand_dims = ms.ops.ExpandDims()

    def construct(self, x):
        """Returns the output probabilities over speakers.
        Arguments
        ---------
        x :
            mindspore tensor.
        """
        for layer in self.blocks:
            x = layer(x)
        # Need to be normalized
        output = self.matmul(self.norm(x), self.norm(self.weight).transpose())
        return output

if __name__ == '__main__':
    input_feats = Tensor(np.ones([1, 32, 60]), ms.float32)
    compute_embedding = ECAPA_TDNN(32, channels=[256, 256, 256, 256, 768], lin_neurons=192)
    outputs = compute_embedding(input_feats)
    print(outputs.shape_)
