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
# ===========================================================================
"""Net Module"""
from mindspore import nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore import Parameter
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
import mindspore.common.initializer as initializer

import src.utils as op


class DAMLoss(nn.Cell):
    """
    Return loss.
    """
    def __init__(self, is_clip=True, clip_value=10):
        super(DAMLoss, self).__init__()
        self.criterion = P.SigmoidCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean()
        self.min_clip_value = P.cast(-clip_value, mstype.float32)
        self.max_clip_value = P.cast(clip_value, mstype.float32)

    def construct(self, logits, labels):
        """Loss Compute Unit"""
        loss = self.criterion(logits, labels)
        loss = self.reduce_mean(P.clip_by_value(loss, self.min_clip_value, self.max_clip_value))
        return loss


class DenseLayer(nn.Cell):
    """
    Full connection layer templates
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True):
        super(DenseLayer, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.reshape = P.Reshape()
        self.shape_op = P.Shape()

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        self.weight = Parameter(initializer.initializer(weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer.initializer(bias_init, [1]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x):
        """Full connection layer"""
        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)
        if self.has_bias:
            x = x + self.bias
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)
        return x


class Attention(nn.Cell):
    '''
    Add attention layer.
    '''
    def __init__(self, attention_type='dot', is_mask=True, mask_value=-2 ** 32 + 1, drop_prob=None):
        super(Attention, self).__init__()

        assert attention_type in ('dot', 'bilinear')
        if attention_type == 'dot':
            self.attention = op.DotSim()
        if attention_type == 'bilinear':
            self.attention = op.BilinearSim()

        self.is_mask = is_mask
        self.mask_value = mask_value
        self.drop_prob = drop_prob
        if self.drop_prob is not None:
            print('use attention dropout')
            self.dropout = nn.Dropout(drop_prob)
        self.softmax = P.Softmax(axis=-1)
        self.weighted_sum = op.BatchMatMulCell()

    def construct(self, Q, K, V, Q_lengths, K_lengths):
        """Attention Compute Unit"""
        Q_time = Q.shape[1]
        K_time = K.shape[1]
        scores = self.attention(Q, K)
        if self.is_mask:
            mask = op.get_mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
            scores = mask * scores + (1 - mask) * self.mask_value
        attention = self.softmax(scores)
        if self.drop_prob is not None:
            attention = self.dropout(attention)
        out = self.weighted_sum(attention, V)

        return out


class FFN(nn.Cell):
    '''
    Add two dense connected layer, max(0, x*W0+b0)*W1+b1.
    '''
    def __init__(self, in_dim, out_dim_0, out_dim_1):
        super(FFN, self).__init__()
        self.fc1 = DenseLayer(in_dim, out_dim_0,
                              weight_init=Tensor(op.orthogonal_init([in_dim, out_dim_0]), dtype=mstype.float32),
                              bias_init='zeros').to_float(mstype.float16)
        self.fc2 = DenseLayer(out_dim_0, out_dim_1,
                              weight_init=Tensor(op.orthogonal_init([out_dim_0, out_dim_1]), dtype=mstype.float32),
                              bias_init='zeros').to_float(mstype.float16)
        self.relu = nn.ReLU()
        self.cast = P.Cast()

    def construct(self, x):
        """Dense Unit"""
        x = self.cast(x, mstype.float16)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        out = self.cast(out, mstype.float32)
        return out


class LayerNormDebug(nn.Cell):
    '''Add layer normalization.

    y=[(x−E[x]) / √(Var[x]+ϵ)]∗ γ + β

    Args:
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''

    def __init__(self, axis=None, shape=200, epsilon=1e-6):
        super(LayerNormDebug, self).__init__()
        if axis is None:
            self.axis = [-1]
        self.shape = [shape]
        self.epsilon = epsilon

        self.scale = Parameter(initializer.initializer("ones", self.shape, mstype.float32))
        self.bias = Parameter(initializer.initializer("zeros", self.shape, mstype.float32))

        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()

    def construct(self, x):
        """LayerNormal Unit"""
        mean = self.reduce_mean(x, self.axis)
        variance = self.reduce_mean(self.square(x - mean), self.axis)
        norm = (x - mean) * P.Rsqrt()(variance + self.epsilon)
        return self.scale * norm + self.bias


class Block(nn.Cell):
    '''
    Add AttentiveBlock Module
    '''

    def __init__(self, in_dim, out_dim_0=None, out_dim_1=None, attention_type='dot',
                 is_layer_norm=True, is_mask=True, mask_value=-2 ** 32 + 1, drop_prob=None):
        super(Block, self).__init__()
        self.attention = Attention(attention_type=attention_type,
                                   is_mask=is_mask,
                                   mask_value=mask_value,
                                   drop_prob=drop_prob)
        if out_dim_0 is None:
            out_dim_0 = in_dim
        if out_dim_1 is None:
            out_dim_1 = out_dim_0

        self.ffn = FFN(in_dim, out_dim_0, out_dim_1)
        self.is_layer_norm = is_layer_norm
        if self.is_layer_norm:
            self.layer_norm1 = LayerNormDebug(shape=in_dim)
            self.layer_norm2 = LayerNormDebug(shape=out_dim_1)

    def construct(self, Q, K, V, Q_lengths, K_lengths):
        """Attention Block Unit"""
        att = self.attention(Q, K, V, Q_lengths, K_lengths)

        if self.is_layer_norm:
            y = self.layer_norm1(att + Q)
        else:
            y = att + Q

        z = self.ffn(y)

        if self.is_layer_norm:
            w = self.layer_norm2(z + y)
        else:
            w = z + y
        return w


class CNN3d(nn.Cell):
    '''Add a 3d convlution layer with relu and max pooling layer.

    x: a tensor with shape [batch_size, 2 * (stack_num+1), max_turn_num, max_turn_len, max_turn_len]

    Returns:
        a flattened tensor with shape [batch, num_features]
    '''

    def __init__(self, in_channels, out_channels_0, out_channels_1, add_relu=True):
        '''

        :param in_channels: 2 * (stack_num + 1)
        :param out_channels_0: ubuntu:32  douban:16
        :param out_channels_1: ubuntu:16  douban:16
        :param add_relu:

        '''
        super(CNN3d, self).__init__()
        self.conv_0 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels_0, kernel_size=3,
                                stride=1, pad_mode='same', has_bias=True,
                                weight_init=initializer.Uniform(0.01), bias_init='zeros')
        self.conv_1 = nn.Conv3d(in_channels=out_channels_0, out_channels=out_channels_1, kernel_size=3,
                                stride=1, pad_mode='same', has_bias=True,
                                weight_init=initializer.Uniform(0.01), bias_init='zeros')
        self.add_relu = add_relu
        if self.add_relu:
            self.elu = nn.ELU()
        self.max_pooling_0 = P.MaxPool3D(kernel_size=3, strides=3, pad_mode="same")
        self.max_pooling_1 = P.MaxPool3D(kernel_size=3, strides=3, pad_mode="same")

    def construct(self, x):
        """3d Conv Unit"""
        x = self.conv_0(x)
        if self.add_relu:
            x = self.elu(x)
        x = self.max_pooling_0(x)

        x = self.conv_1(x)
        if self.add_relu:
            x = self.elu(x)
        x = self.max_pooling_1(x)
        return x
