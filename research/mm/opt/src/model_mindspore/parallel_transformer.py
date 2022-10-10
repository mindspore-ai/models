# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Transformer Networks"""

import math


import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import nn
from mindspore._checkparam import Validator
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.seed import _get_graph_seed
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.nn import Cell
from mindspore.nn.layer import Dense
from mindspore.ops import constexpr
from mindspore.ops import functional as F
from mindspore.ops import operations as P

__all__ = [
    "Dropout",
    "LayerNorm",
    "Linear",
    "VocabEmbedding",
    "MultiHeadAttention",
    "FeedForwardCell",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderCell",
    "TransformerDecoderCell",
    "Transformer",
    "position_encoding",
    "ParallelConfig"]


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    r"""
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        depth (int): Hidden size.
        min_timescale (float): Default: 1.
        max_timescale (float): Default: 10000.

    Returns:
        Tensor of shape (length, depth)
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class ParallelConfig:
    r"""
        ParallelConfig for the setting the global data parallel, model parallel and fusion group.
    """
    dp = 16
    mp = 1
    pipeline_stage = 1
    recompute = False
    optimizer_shard = True
    fusion_group = 24
    parallel_mode = ParallelMode.SEMI_AUTO_PARALLEL
    vocab_emb_dp = True
    ep = dp
    capacity_factor = 1.5
    expert_num = 32
    aux_loss_factor = 0.01

    @staticmethod
    def set_global_parallel_config(dp=1,
                                   mp=1,
                                   recompute=True,
                                   stages=1,
                                   optimizer_shard=True,
                                   fusion_group=4,
                                   parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                   vocab_emb_dp=True):
        r"""
        The parallel configure setting

        Args:
            dp (int): The data parallel way. Default: 1
            mp (int): The model parallel way. Default: 1
            stages (int): The number of the pipeline stage. Should be a positive value. Default: 1.
            optimizer_shard (bool): Enable optimizer state sharding or not. Default: True.
            fusion_group (int): The fusion group size of the optimizer state sharding. Default: 4.
            recompute (bool): Enable recomputation of the transformer block or not. Default: False.
            parallel_mode (ParallelMode): Can be SEMI_AUTO_PARALLEL, DATA_AUTO_PARALLEL or AUTO_PARALLEL.
            vocab_emb_dp (bool): Shard embedding in model parallel or data parallel. Default: True

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> ParallelConfig(dp=1, mp=1)
            >>> ParallelConfig(stages=4)
        """
        ParallelConfig.dp = dp
        ParallelConfig.mp = mp
        ParallelConfig.pipeline_stage = stages
        ParallelConfig.optimizer_shard = optimizer_shard
        ParallelConfig.fusion_group = fusion_group
        ParallelConfig.recompute = recompute
        ParallelConfig.parallel_mode = parallel_mode
        ParallelConfig.vocab_emb_dp = vocab_emb_dp


class Dropout(Cell):
    r"""
        A Dropout Implements with P.DropoutGenMask and  P.DropoutDoMask for parallel training.
        Args:
            keep_prob: the keep probability of the inputs. Default 0.5
            dtype: the input type. Default mstype.float32

        Inputs:
            x: To be dropped tensor.

        Returns:
            a tensor with dropped value.
        Examples:
            >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
            >>> net = nn.Dropout(keep_prob=0.8)
            >>> net.set_train()
            Dropout<keep_prob=0.8>
            >>> output = net(x)
            >>> print(output.shape)
            (2, 2, 3)
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "dropout probability should be a number in range (0, 1], but got {}".format(
                    keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        self.is_ascend = context.get_context('device_target') in ["Ascend"]
        if self.is_ascend:
            seed0, seed1 = _get_graph_seed(0, "dropout")
            self.seed0 = seed0
            self.seed1 = seed1
            self.dtype = dtype
            self.get_shape = P.Shape()
            self.dropout_gen_mask = P.DropoutGenMask(Seed0=self.seed0, Seed1=self.seed1)
            self.dropout_do_mask = P.DropoutDoMask()
            self.cast = P.Cast()
        else:
            self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        """Construct method for Dropout"""
        if not self.training:
            return x

        if not self.is_ascend:
            out, _ = self.dropout(x)
            return out

        if self.keep_prob == 1:
            return x

        shape = self.get_shape(x)
        dtype = P.DType()(x)
        keep_prob = self.cast(self.keep_prob, dtype)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        return 'keep_prob={}, dtype={}'.format(self.keep_prob, self.dtype)

    def shard(self, strategy):
        r"""
        Set the shard for the dropout. the strategy size should be equal to the inputs.

        Args:
            strategy (tuple): The strategy for the dropout. Should be the same shape as the inputs.
        Examples:
            >>> net = nn.Dropout(keep_prob=0.8)
            >>> net.set_train()
            Dropout<keep_prob=0.8>
            >>> net.shard(((2, 1),))
        """
        if self.is_ascend:
            self.dropout_gen_mask.shard(strategy)
            self.dropout_do_mask.shard(strategy)
        else:
            self.dropout.shard(strategy)
        return self


class LayerNorm2(Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            dp (int): The data parallel way of the inputs, Default:1
            eps (float): The epsilon value of the denominator. Default 1e-5.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, dp=1, eps=1e-5):
        super(LayerNorm2, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma", parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta", parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1),))
        self.square = P.Square().shard(((dp, 1),))
        self.sqrt = P.Sqrt().shard(((dp, 1),))
        self.sub1 = P.Sub().shard(((dp, 1), (dp, 1)))
        self.sub2 = P.Sub().shard(((dp, 1), (dp, 1)))
        self.add = P.TensorAdd().shard(((dp, 1), ()))
        self.eps = eps
        self.mul = P.Mul().shard(((dp, 1), (1,)))
        self.add2 = P.TensorAdd().shard(((dp, 1), (1,)))
        self.real_div = P.RealDiv().shard(((dp, 1), (dp, 1)))

    def construct(self, x):
        r"""
          x : batch x seq_length x hidden_size
        """
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class LayerNorm(Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            dp (int): The data parallel way of the inputs, Default:1
            eps (float): The epsilon value of the denominator. Default 1e-5.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, dp=1, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma", parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta", parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.square = P.Square().shard(((dp, 1, 1),))
        self.sqrt = P.Sqrt().shard(((dp, 1, 1),))
        self.sub1 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.sub2 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add = P.TensorAdd().shard(((dp, 1, 1), ()))
        self.eps = eps
        self.mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.add2 = P.TensorAdd().shard(((dp, 1, 1), (1,)))
        self.real_div = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))

    def construct(self, x):
        r"""
          x : batch x seq_length x hidden_size
        """
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class Linear(Dense):
    r"""
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    3-D tensor.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    where :math:`X` is the input tensors, :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: None.
        compute_dtype (mstype): The computation type. Default: mstype.float16
        shard_output (bool): The sharding of the `matmul` operation. If true, the shard will focus on the relative
            dimension and the output will not be sharded. Ortherwise will shard on the output dimension to mp dimension.
        parallel_config (ParallelConfig): The parallel configuration. Default: None
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.ones((10, 20, 3)), mindspore.float32)
        >>> net = Linear(3, 4)
        >>> output = net(x)
        >>> print(output.shape)
        (10, 20, 4)
    """

    @cell_attr_register(attrs=['has_bias', 'in_channels', 'out_channels', 'shard_output', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 compute_dtype=mstype.float16,
                 shard_output=True,
                 parallel_config=ParallelConfig):
        super(Linear, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     weight_init=weight_init,
                                     bias_init=bias_init,
                                     has_bias=has_bias,
                                     activation=activation)
        if activation and not isinstance(activation, str):
            raise ValueError("Activation can only be str, but found type {}".format(activation))
        self.act_name = activation
        self.dtype = compute_dtype
        self.cast = P.Cast()
        if not parallel_config.optimizer_shard:
            self.weight.parallel_optimizer = False
        if has_bias:
            self.bias.parallel_optimizer = False
        if shard_output:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.dp, 1), (parallel_config.mp, 1)))
            self.add = P.TensorAdd().shard(((parallel_config.dp, parallel_config.mp), (parallel_config.mp,)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(
                ((parallel_config.dp, parallel_config.mp), (1, parallel_config.mp)))
            self.add = P.TensorAdd().shard(((parallel_config.dp, 1), (1,)))
        if self.activation_flag:
            getattr(self.activation, self.act_name).shard(((parallel_config.dp, 1, parallel_config.mp),))

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        if self.activation_flag:
            output = self.activation(output)
        return output


class FeedForwardCell(Cell):
    """
    The multilayer perceptron with two linear layers with dropout applied at final output. The first linear
    will project the input dimension from hidden_size to ffn_hidden_size, the second linear will project the
    dimension from ffn_hidden_size to hidden_size. The first linear is sharded on the relative dimension,
    the second linear is sharded on the output dimension.
    Args:
        hidden_size (int): The dimension of the inputs.
        ffn_hidden_size (int): The intermediate hidden size.
        dropout_rate (float): The dropout rate for the second linear's output.
        hidden_act (str): The activate type of the first linear, Default: gelu.
        config(ParallelConfig): the config of parallel setting, see `ParallelConfig`
    Inputs:
        x: should be `[batch, seq_length, hidden_size]`.
    Returns:
        output: Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size]`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = FeedForwardCell(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1)
        >>> tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
        >>> output = model(tensor)
    """

    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 parallel_config=ParallelConfig):
        super(FeedForwardCell, self).__init__()
        input_size = hidden_size
        output_size = ffn_hidden_size
        # Project to ffn_hidden_size
        self.mapping = Linear(in_channels=input_size,
                              out_channels=output_size,
                              parallel_config=parallel_config,
                              activation=hidden_act,
                              shard_output=True)
        # Project back to embedding_size
        self.projection = Linear(in_channels=output_size,
                                 out_channels=input_size,
                                 parallel_config=parallel_config,
                                 shard_output=False)
        self.dropout = Dropout(1 - dropout_rate)
        self.dropout.shard(((parallel_config.dp, 1, 1),))
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        # [bs, seq_length, ffn_hidden_size]
        hidden = self.mapping(x)
        output = self.projection(hidden)
        # [bs, seq_length, hidden_size]
        output = self.dropout(output)
        return output, 0


class AttentionMask(Cell):
    r"""
    Get the Lower triangular matrix.
    Args:
        seq_length: the length of the
        config(parallel_config): the parallel configure
    Inputs:
        input_mask: the mask indicating whether each position is a valid input with (batch_size, seq_length)
    Outputs:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, seq_length, parallel_config=ParallelConfig):
        super(AttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        self.expand_dim_3 = P.ExpandDims().shard(((parallel_config.dp, 1, 1),))
        ones = np.ones(shape=(seq_length, seq_length))
        # Default lower triangle mask matrix
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        r"""
        Generate the attention mask matrix.
        """
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        # [bs, seq_length, seq_length]
        attention_mask = self.multiply(
            attention_mask, lower_traiangle)
        return self.expand_dim_3(attention_mask, 1)


class BertAttentionMask(Cell):
    r"""
    Get the Lower triangular matrix.
    Args:
        seq_length: the length of the
        config(parallel_config): the parallel configure
    Inputs:
        input_mask: the mask indicating whether each position is a valid input with (batch_size, seq_length)
    Outputs:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, seq_length, parallel_config=ParallelConfig):
        super(BertAttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        self.expand_dim_3 = P.ExpandDims().shard(((parallel_config.dp, 1, 1),))
        # ones = np.ones(shape=(seq_length, seq_length))
        # Default lower triangle mask matrix
        self.multiply = P.Mul().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        r"""
        Generate the attention mask matrix.
        """
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        # [bs, seq_length, seq_length]
        return self.expand_dim_3(attention_mask, 1)


class BertAttentionMaskWithoutLen(Cell):
    r"""
    Get the Lower triangular matrix.
    Args:
        seq_length: the length of the
        config(parallel_config): the parallel configure
    Inputs:
        input_mask: the mask indicating whether each position is a valid input with (batch_size, seq_length)
    Outputs:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, parallel_config=ParallelConfig):
        super(BertAttentionMaskWithoutLen, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        self.expand_dim_3 = P.ExpandDims().shard(((parallel_config.dp, 1, 1),))
        # Default lower triangle mask matrix
        self.multiply = P.Mul().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        r"""
        Generate the attention mask matrix.
        """
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        # [bs, seq_length, seq_length]
        return self.expand_dim_3(attention_mask, 1)


class VocabEmbedding(Cell):
    """
    The embedding lookup table for vocabulary
    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        config(ParallelConfig): the parallel config of network.
    Inputs:
        input_ids: the tokenized inputs with datatype int32 with shape (batch_size, seq_length)
    Outputs:
        output: Tensor, the embedding vector for the input with shape (batch_size,
        seq_length, embedding_size)
        self.weight: Tensor, the embedding table for the vocabulary

    Raises:
        ValueError: If the ParallelConfig.vocab_emb_dp is True, the vocab size is not a multiple of ParallelConfig.mp
    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = VocabEmbedding(vocab_size=30, embedding_size=30)
        >>> tensor = Tensor(np.ones((20, 15)), dtype.int32)
        >>> output = model(tensor)
    """

    def __init__(self, vocab_size, embedding_size, parallel_config=ParallelConfig, param_init='normal'):
        super(VocabEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_table = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                         name='embedding_table', parallel_optimizer=False)
        if parallel_config.vocab_emb_dp:
            self.gather = P.GatherV2().shard(((1, 1), (parallel_config.dp, 1)))
        else:
            if self.embedding_size % parallel_config.mp != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of ParallelConfig.mp {ParallelConfig.mp}.")
            self.gather = P.GatherV2().shard(((parallel_config.mp, 1), (1, 1)))

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table.value()


class MultiHeadAttention(Cell):
    """
    MultiHeadAttention module.

    Args:
        hidden_size(int): The hidden size of the input.
        from_seq_length(int): The seq_length of the query tensor.
        to_seq_length(int): The seq_length of the key and value tensor.
        num_heads(int): The number of the heads.
        hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1
        attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1
        compute_dtype(mstype): The computation type. Default mstype.float16. The computation of the
            softmax will be converted to the float32.
        use_past(bool): Use the past state to compute. Default False.
        parallel_config(ParallelConfig): The parallel configure.
    Inputs:
        from_tensor: the query vector with shape (batch_size, src_seq_length, hidden_size).
        to_tensor: the key and value vector with shape (batch_size, tgt_seq_length, hidden_size).
        attention_mask: the attention mask matrix with shape (batch_size, 1,
        seq_length, seq_length)
        layer_past: the previous feature map

    Outputs:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer

    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = MultiHeadAttention(hidden_size=15, from_seq_length=20, to_seq_length=20,
        >>>                           num_heads=3)
        >>> from_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
        >>> to_tensor = Tensor(np.ones((2, 20, 15)), dtype.float16)
        >>> attention_mask = Tensor(np.ones((2, 1, 20, 20)), dtype.float16)
        >>> model(from_tensor, to_tensor, attention_mask)
    """

    def __init__(self, hidden_size,
                 from_seq_length,
                 to_seq_length,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 parallel_config=ParallelConfig):
        super(MultiHeadAttention, self).__init__()
        # Output layer
        self.projection = Linear(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 compute_dtype=compute_dtype,
                                 parallel_config=parallel_config,
                                 shard_output=False)
        self.transpose = P.Transpose().shard(((parallel_config.dp, 1, parallel_config.mp, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.dp, parallel_config.mp, 1, 1),))
        self.reshape = P.Reshape()
        self.n_head = num_heads
        # embedding size per head
        self.size_per_head = hidden_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=mstype.float32)
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.dp, parallel_config.mp, 1, 1), (parallel_config.dp, parallel_config.mp, 1, 1)))
        self.real_div = P.RealDiv().shard(((parallel_config.dp, parallel_config.mp, 1, 1), ()))
        self.sub = P.Sub().shard(
            ((1,), (parallel_config.dp, 1, 1, 1)))
        self.mul = P.Mul().shard(
            ((parallel_config.dp, 1, 1, 1), (1,)))
        self.add = P.TensorAdd().shard(
            ((parallel_config.dp, 1, 1, 1), (parallel_config.dp, parallel_config.mp, 1, 1)))
        # Normalize factor for attention, sqrt(dk) as widely used
        self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        self.use_past = use_past
        self.dropout = Dropout(1 - hidden_dropout_rate)
        self.dropout.shard(((parallel_config.dp, 1, 1),))
        self.prob_dropout = Dropout(1 - attention_dropout_rate)
        self.prob_dropout.shard(
            ((parallel_config.dp, parallel_config.mp, 1, 1),))
        self.softmax = nn.Softmax()
        self.softmax.softmax.shard(((parallel_config.dp, parallel_config.mp, 1),))
        self.expand_dims = P.ExpandDims().shard(((parallel_config.dp, 1, 1),))

        # Query
        self.dense1 = Linear(hidden_size,
                             hidden_size,
                             parallel_config=parallel_config,
                             shard_output=True).to_float(compute_dtype)
        # Key
        self.dense2 = Linear(hidden_size,
                             hidden_size,
                             parallel_config=parallel_config,
                             shard_output=True).to_float(compute_dtype)
        # Value
        self.dense3 = Linear(hidden_size,
                             hidden_size,
                             parallel_config=parallel_config,
                             shard_output=True).to_float(compute_dtype)

        self.stack = P.Stack().shard(((parallel_config.dp, parallel_config.mp, 1, 1),
                                      (parallel_config.dp, parallel_config.mp, 1, 1)))

    def construct(self, from_tensor, to_tensor, attention_mask, layer_past=None):
        """
        multi-head attention
        """

        from_tensor_original_shape = F.shape(from_tensor)
        from_tensor = F.reshape(from_tensor, (-1, from_tensor_original_shape[-1]))

        to_tensor_original_shape = F.shape(to_tensor)
        to_tensor = F.reshape(to_tensor, (-1, to_tensor_original_shape[-1]))

        # Self attention: query, key, value are derived from the same inputs
        query = self.dense1(from_tensor)
        key = self.dense2(to_tensor)
        value = self.dense3(to_tensor)
        # [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (-1, from_tensor_original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # [bs, num_heads, size_per_head, seq_length]
        key = self.transpose(
            F.reshape(
                key, (-1, to_tensor_original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (-1, to_tensor_original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        layer_present = (key, value)
        # attention considering attention mask
        attention = self._attn(query, key, value, attention_mask)
        # [bs, seq_length, embedding_size]
        attention_merge = self.merge_heads(attention)
        # Output
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present

    def split_heads(self, x, transpose):
        """
        split 3d tensor to 4d and switch certain axes
        Inputs:
            x: input tensor
            transpose: tuple, the transpose sequence
        Outputs:
            x_transpose: the 4d output
        """
        x_size = P.Shape()(x)
        new_x_shape = x_size[:-1] + (self.n_head, self.size_per_head)
        x = self.reshape(x, new_x_shape)
        x_transpose = self.transpose(x, transpose)
        return x_transpose

    def merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 3d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = x_shape[:-2] + (x_shape[-2] * x_shape[-1],)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        score = self.batch_matmul(query, key)
        # Normalize after query and key MatMul
        score = self.real_div(
            score,
            P.Cast()(self.scale_factor, P.DType()(score)))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, mstype.float32)
        # Minus 10000 for the position where masked to exclude them from softmax
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        shape = F.shape(attention_scores)
        # attention probs
        attention_probs = self.softmax(
            F.reshape(attention_scores,
                      (shape[0], -1, shape[-1])))
        attention_probs = P.Cast()(attention_probs, ori_dtype)
        attention_probs = F.reshape(attention_probs, shape)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values


class TransformerEncoderCell(Cell):
    r"""
    Transformer Encoder module.

    Args:
        hidden_size(int): The hidden size of the input.
        ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
        num_heads(int): The number of the heads.
        hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1
        attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1
        post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
        hidden_act(str): The activation of the internal feedforward layer. Default 'gelu'.
        parallel_config(ParallelConfig): The parallel configure.
    Inputs:
        x: Tensor, shape should be [batch_size, seq_length, hidden_size]
        input_mask: Tensor, attention mask with shape [batch_size, 1, seq_length, seq_length]
        layer_past: the past the feature map.
    Outputs:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer

    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = TransformerEncoderCell(hidden_size=8, ffn_hidden_size=64, seq_length=16,
        >>>                                 num_heads=2)
        >>> encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
        >>> encoder_input_mask = Tensor(np.ones((2, 1, 16, 16)), dtype.float16)
        >>> model(encoder_input_value, encoder_input_value)
    """

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 hidden_act='gelu',
                 use_moe=False,
                 parallel_config=ParallelConfig):
        super(TransformerEncoderCell, self).__init__()
        if num_heads % parallel_config.mp != 0:
            raise ValueError(
                f"num heads must be divisibled by the model parallel way {parallel_config.mp}, but found {num_heads}")

        self.layernorm1 = LayerNorm((hidden_size,), parallel_config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((hidden_size,), parallel_config.dp).to_float(mstype.float32)

        self.attention = MultiHeadAttention(hidden_size=hidden_size,
                                            from_seq_length=seq_length,
                                            to_seq_length=seq_length,
                                            num_heads=num_heads,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            parallel_config=parallel_config)

        if use_moe:
            self.output = MoE(in_channels=hidden_size,
                              config=parallel_config,
                              hidden_size=ffn_hidden_size,
                              out_channels=hidden_size,
                              hidden_act=hidden_act)
        else:
            # Feed Forward Network, FFN
            self.output = FeedForwardCell(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
        self.post_layernorm_residual = post_layernorm_residual
        self.add = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.dtype = mstype.float16

    def construct(self, x, input_mask, layer_past=None):
        r"""
        The forward process of the block.
        """
        # [bs, seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x, input_x, input_mask,
                                                  layer_past)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit, moe_loss = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present, moe_loss


class TransformerDecoderCell(Cell):
    r"""
    Transformer Decoder module.

    Args:
        hidden_size(int): The hidden size of the input.
        ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
        num_heads(int): The number of the heads.
        hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1
        attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1
        post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
        hidden_act(str): The activation of the internal feedforward layer. Default 'gelu'.
        parallel_config(ParallelConfig): The parallel configure.
    Inputs:
        hidden_stats: the input tensor with shape [batch_size, seq_length, hidden_size]
        decoder_mask: the attention mask for decoder with shape [batch_size, 1, seq_length, seq_length]
        encoder_output: the output of the encoder with shape [batch_size, seq_length, hidden_size]
        memory_mask: the memory mask of the cross attention with shape [batch, 1, tgt_seq_length, src_seq_length]
         where tgt_seq_length is the length of the decoder.
        layer_past: the past the feature map.
    Outputs:
        output: Tensor, the output logit of this layer. The shape is [batch, seq_length, hidden_size]
        layer_present: Tensor, the feature map of current layer
    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = TransformerDecoderCell(hidden_size=64, ffn_hidden_size=64, num_heads=2, seq_length=10)
        >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
        >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
        >>> decoder_input_mask = Tensor(np.ones((2, 1, 10, 10)), dtype.float16)
        >>> memory_mask = Tensor(np.ones((2, 1, 10, 20)), dtype.float16)
        >>> model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)
    """

    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 hidden_act='gelu',
                 use_moe=False,
                 parallel_config=ParallelConfig):
        super(TransformerDecoderCell, self).__init__()
        if num_heads % parallel_config.mp != 0:
            raise ValueError(
                f"num heads must be divisibled by the model parallel way {parallel_config.mp}, but found {num_heads}")

        self.layernorm1 = LayerNorm((hidden_size,), parallel_config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((hidden_size,), parallel_config.dp).to_float(mstype.float32)

        self.attention = MultiHeadAttention(hidden_size=hidden_size,
                                            from_seq_length=seq_length,
                                            to_seq_length=seq_length,
                                            num_heads=num_heads,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            parallel_config=parallel_config)
        # Cross attention with the output of encoder as memory tensor
        self.cross_attention = MultiHeadAttention(hidden_size=hidden_size,
                                                  from_seq_length=seq_length,
                                                  to_seq_length=seq_length,
                                                  num_heads=num_heads,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  parallel_config=parallel_config)
        self.cross_attention_layernorm = LayerNorm((hidden_size,), parallel_config.dp).to_float(mstype.float32)

        if use_moe:
            self.output = MoE(in_channels=hidden_size,
                              config=parallel_config,
                              hidden_size=ffn_hidden_size,
                              out_channels=hidden_size,
                              hidden_act=hidden_act)

        else:
            # Feed Forward Network, FFN
            self.output = FeedForwardCell(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)

        self.post_layernorm_residual = post_layernorm_residual
        self.add = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.dtype = mstype.float16

    def construct(self, hidden_stats,
                  decoder_mask,
                  encoder_output,
                  memory_mask,
                  layer_past=None):
        r"""
        The forward process of the block.
        """
        # [bs, seq_length, embedding_size]
        input_x = self.layernorm1(hidden_stats)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x, input_x, decoder_mask, layer_past)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(hidden_stats, attention)

        middle_output = self.cross_attention_layernorm(x)
        middle_output = F.cast(middle_output, self.dtype)
        cross_attn_output, layer_present = self.cross_attention(middle_output, encoder_output,
                                                                memory_mask, layer_past)
        if self.post_layernorm_residual:
            x = self.add(middle_output, cross_attn_output)
        else:
            x = self.add(x, cross_attn_output)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit, moe_loss = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present, moe_loss


def set_parallel_configure_for_layer(network, layer_id, offset, layers, parallel_config):
    # Used for the pipeline's stages setting
    network.pipeline_stage = (layer_id + offset) * parallel_config.pipeline_stage // layers
    # Used for optimizer's fusion tag
    layer_per_group = layers / parallel_config.fusion_group
    network.set_comm_fusion(int(layer_id / layer_per_group + 1) + offset)
    # Used for enabling recomputation of the block
    if parallel_config.recompute:
        network.recompute()


class TransformerEncoder(Cell):
    r"""
    Transformer Encoder module with multi-layer.

    Args:
        num_layers(int): The layers of the `TransformerEncoderCell`
        hidden_size(int): The hidden size of the input.
        ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
        seq_length(int): The seq_length of the input tensor.
        num_heads(int): The number of the heads.
        hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1
        attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1
        post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
        hidden_act(str): The activation of the internal feedforward layer. Default 'gelu'.
        parallel_config(ParallelConfig): The parallel configure.
        lambda_func: a function can specific the fusion index, pipeline stages and recompute attribute.
            Default: set_parallel_configure_for_layer
        offset(int): The initial layer index for the `decoder`. Used for setting the fusion id and stage id, to not
            overlap with the encoder layer.
    Inputs:
        hidden_states: Tensor, shape should be [batch_size, seq_length, hidden_size]
        attention_mask: Tensor, attention mask with shape [batch_size, 1, seq_length, seq_length]
        layer_past: the past the feature map.
    Outputs:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> model = TransformerEncoder(num_layers=2, hidden_size=8, ffn_hidden_size=64, seq_length=16,
        >>>                       num_heads=2)
        >>> encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
        >>> encoder_input_mask = Tensor(np.ones((2, 1, 16, 16)), dtype.float16)
        >>> model(encoder_input_value, encoder_input_mask)
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 parallel_config=ParallelConfig,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 use_moe=False,
                 lambda_func=set_parallel_configure_for_layer,
                 offset=0):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.CellList()
        self.accu_loss = Tensor(0.0, mstype.float32)
        self.add = P.Add().shard(((), ()))
        for i in range(num_layers):
            block = TransformerEncoderCell(hidden_size=hidden_size,
                                           ffn_hidden_size=ffn_hidden_size,
                                           seq_length=seq_length,
                                           attention_dropout_rate=attention_dropout_rate,
                                           hidden_dropout_rate=hidden_dropout_rate,
                                           num_heads=num_heads,
                                           hidden_act=hidden_act,
                                           post_layernorm_residual=post_layernorm_residual,
                                           use_moe=use_moe,
                                           parallel_config=parallel_config)
            if lambda_func:
                lambda_func(block, layer_id=i, offset=offset,
                            layers=num_layers, parallel_config=parallel_config)
            self.blocks.append(block)

    def construct(self, hidden_states, attention_mask, layer_past=None):
        r"""
        The forward process of the block.
        """
        present_layer = ()
        accu_loss = self.accu_loss
        for i in range(self.num_layers):
            hidden_states, present, moe_loss = self.blocks[i](hidden_states,
                                                              attention_mask,
                                                              layer_past)
            present_layer = present_layer + (present,)
            accu_loss = self.add(accu_loss, moe_loss)

        return hidden_states, present_layer, accu_loss


class TransformerDecoder(Cell):
    r"""
    Transformer Decoder module with multi-layer.

    Args:
        num_layers(int): The layers of the `TransformerEncoderCell`
        hidden_size(int): The hidden size of the input.
        ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
        seq_length(int): The seq_length of the input tensor.
        num_heads(int): The number of the heads.
        hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1
        attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1
        post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
        hidden_act(str): The activation of the internal feedforward layer. Default 'gelu'.
        parallel_config(ParallelConfig): The parallel configure.
                lambda_func: a function can specific the fusion index, pipeline stages and recompute attribute.
                Default: set_parallel_configure_for_layer
        offset(int): The initial layer index for the `decoder`. Used for setting the fusion id and stage id, to not
            overlap with the encoder layer.
    Inputs:
        hidden_stats: the input tensor with shape [batch_size, seq_length, hidden_size]
        attention_mask: the attention mask for decoder with shape [batch_size, 1, seq_length, seq_length]
        encoder_output: the output of the encoder with shape [batch_size, seq_length, hidden_size]
        memory_mask: the memory mask of the cross attention with shape [batch, 1, tgt_seq_length, src_seq_length]
         where tgt_seq_length is the length of the decoder. the output of the encoder with shape
         [batch_size, seq_length, hidden_size],
        layer_past: the past the feature map.
    Outputs:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = TransformerDecoder(num_layers=1, hidden_size=64, ffn_hidden_size=64, num_heads=2, seq_length=10)
        >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
        >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
        >>> decoder_input_mask = Tensor(np.ones((2, 1, 10, 10)), dtype.float16)
        >>> memory_mask = Tensor(np.ones((2, 1, 10, 20)), dtype.float16)
        >>> model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 parallel_config=ParallelConfig,
                 post_layernorm_residual=False,
                 hidden_act='gelu',
                 use_moe=False,
                 lambda_func=set_parallel_configure_for_layer,
                 offset=0):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.CellList()
        self.accu_loss = Tensor(0.0, mstype.float32)
        self.add = P.Add().shard(((), ()))
        for i in range(num_layers):
            block = TransformerDecoderCell(hidden_size=hidden_size,
                                           ffn_hidden_size=ffn_hidden_size,
                                           seq_length=seq_length,
                                           attention_dropout_rate=attention_dropout_rate,
                                           hidden_dropout_rate=hidden_dropout_rate,
                                           num_heads=num_heads,
                                           hidden_act=hidden_act,
                                           post_layernorm_residual=post_layernorm_residual,
                                           use_moe=use_moe,
                                           parallel_config=parallel_config)

            # Used for the pipeline's stages setting
            if lambda_func:
                lambda_func(block, layer_id=i, offset=offset,
                            layers=num_layers, parallel_config=parallel_config)
            self.blocks.append(block)

    def construct(self, hidden_states, attention_mask, encoder_output, memory_mask, layer_past=None):
        r"""
        The forward process of the block.
        """
        present_layer = ()
        accu_loss = self.accu_loss
        # Loop through each self-attention layer
        for i in range(self.num_layers):
            hidden_states, present, moe_loss = self.blocks[i](hidden_states,
                                                              attention_mask,
                                                              encoder_output,
                                                              memory_mask,
                                                              layer_past)
            present_layer = present_layer + (present,)
            accu_loss = self.add(accu_loss, moe_loss)

        return hidden_states, present_layer, accu_loss


class Transformer(Cell):
    r"""
    Transformer Decoder module.

    Args:
        encoder_layers(int): The layers of the `TransformerEncoderCell`
        decoder_layers(int): The layers of the `TransformerDecoderCell`
        hidden_size(int): The hidden size of the input.
        ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
        src_seq_length(int): The seq_length of the encoder's input tensor.
        tgt_seq_length(int): The seq_length of the decoder's input tensor.
        num_heads(int): The number of the heads.
        hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1
        attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1
        post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
        hidden_act(str): The activation of the internal feedforward layer. Default 'gelu'.
        parallel_config(ParallelConfig): The parallel configure.
    Inputs:
        encoder_inputs: the input tensor with shape [batch_size, seq_length, hidden_size]
        encoder_masks: the attention mask for decoder with shape [batch_size, 1, seq_length, seq_length]
        decoder_inputs: the output of the encoder with shape [batch_size, seq_length, hidden_size], this can be none if
            the decoder layer is 0.
        decoder_masks: the attention mask for decoder with shape [batch_size, 1, seq_length, seq_length]
        memory_mask: the memory mask of the cross attention with shape [batch, 1, tgt_seq_length, src_seq_length]
         where tgt_seq_length is the length of the decoder. the output of the encoder with shape [batch_size,
         seq_length, hidden_size], this can be none if the decoder layer is 0.
    Outputs:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> model = Transformer(encoder_layers=1, decoder_layers=2, hidden_size=64, ffn_hidden_size=64, \
        >>>      src_seq_length=20, tgt_seq_length=20)
        >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
        >>> encoder_input_mask = Tensor(np.ones((2, 1, 20, 20)), dtype.float16)
        >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
        >>> decoder_input_mask = Tensor(np.ones((2, 1, 10, 10)), dtype.float16)
        >>> memory_mask = Tensor(np.ones((2, 1, 10, 20)), dtype.float16)
        >>> model(encoder_input_value, encoder_input_mask, decoder_input_value, decoder_input_mask, \
        >>>              memory_mask)
    """

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 src_seq_length,
                 tgt_seq_length,
                 encoder_layers=3,
                 decoder_layers=3,
                 num_heads=2,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 parallel_config=ParallelConfig,
                 use_moe=False,
                 post_layernorm_residual=False):
        super(Transformer, self).__init__()

        self.accu_loss = Tensor(0.0, mstype.float32)
        self.add = P.Add().shard(((), ()))
        # The shard setting of Transformer is set within the class StackedTransformer
        if encoder_layers > 0:
            self.encoder = TransformerEncoder(num_layers=encoder_layers,
                                              hidden_size=hidden_size,
                                              ffn_hidden_size=ffn_hidden_size,
                                              num_heads=num_heads,
                                              seq_length=src_seq_length,
                                              attention_dropout_rate=attention_dropout_rate,
                                              hidden_dropout_rate=hidden_dropout_rate,
                                              hidden_act=hidden_act,
                                              post_layernorm_residual=post_layernorm_residual,
                                              use_moe=use_moe,
                                              parallel_config=parallel_config)
        else:
            self.encoder = None

        # Offset is needed as the encoder has consumed some flags.
        # so the decoder need to increase the flags based on the encoder layer
        if decoder_layers > 0:
            self.decoder = TransformerDecoder(num_layers=decoder_layers,
                                              hidden_size=hidden_size,
                                              ffn_hidden_size=ffn_hidden_size,
                                              parallel_config=parallel_config,
                                              num_heads=num_heads,
                                              seq_length=tgt_seq_length,
                                              attention_dropout_rate=attention_dropout_rate,
                                              hidden_dropout_rate=hidden_dropout_rate,
                                              hidden_act=hidden_act,
                                              post_layernorm_residual=post_layernorm_residual,
                                              use_moe=use_moe,
                                              offset=encoder_layers)
        else:
            self.decoder = None

    def construct(self, encoder_inputs,
                  encoder_masks,
                  decoder_inputs=None,
                  decoder_masks=None,
                  memory_mask=None):
        """
        construct
        :param encoder_inputs:
        :param encoder_masks:
        :param decoder_inputs:
        :param decoder_masks:
        :param memory_mask:
        :return:
        """


        encoder_output = None
        output = None
        encoder_layer_present = None
        decoder_layer_present = None
        accu_loss = self.accu_loss
        if self.encoder is not None:
            encoder_output, encoder_layer_present, moe_loss_enc = self.encoder(encoder_inputs, encoder_masks)
            output = encoder_output
            accu_loss = self.add(accu_loss, moe_loss_enc)

        if self.decoder is not None:
            # decoder mask can be created outside of the model
            decoder_output, decoder_layer_present, moe_loss_dec = self.decoder(decoder_inputs,
                                                                               decoder_masks,
                                                                               encoder_output,
                                                                               memory_mask)
            output = decoder_output
            accu_loss = self.add(accu_loss, moe_loss_dec)

        return output, encoder_layer_present, decoder_layer_present, accu_loss


class CumSum(nn.Cell):
    """CumSum"""

    def __init__(self, config):
        super(CumSum, self).__init__()
        self.range = P.Range().shard(((1,),))
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(((config.dp, 1), (1, 1)))
        self.shape = P.Shape()
        self.cast = P.Cast()

        self.transpose = P.Transpose().shard(((config.dp, 1, 1),))
        self.transpose2 = P.Transpose().shard(((1, 1),))
        self.transpose3 = P.Transpose().shard(((config.dp, 1, 1),))
        self.expand = P.ExpandDims().shard(((1,),))
        self.greater = P.Greater().shard(((1, 1), (1, 1)))

        self.start = Tensor(0, mstype.int32)
        self.limit = Tensor(0, mstype.int32)
        self.delta = Tensor(1, mstype.int32)
        self.add = P.TensorAdd().shard(((1,), ()))

    def construct(self, expert_mask, tokens_per_device):
        """Construct method"""
        origin_shape = self.shape(expert_mask)
        expert_mask_trans = self.transpose(expert_mask, (0, 2, 1))
        expert_mask_reshaped = self.reshape(expert_mask_trans, (-1, tokens_per_device))

        one_dim = self.expand(self.range(self.start, self.add(self.limit, tokens_per_device), self.delta), 0)
        other_dim = self.transpose2(one_dim, (1, 0))
        up_tri_matrix = self.greater(one_dim, other_dim)
        up_tri_matrix = self.cast(up_tri_matrix, mstype.float32)

        cum_sum = self.matmul(expert_mask_reshaped, up_tri_matrix)
        cum_sum = self.reshape(cum_sum, (origin_shape[0], origin_shape[2], tokens_per_device))
        cum_sum = self.transpose3(cum_sum, (0, 2, 1))
        return cum_sum


@constexpr
def caculate_expert_capacity(tokens_per_device, capacity_factor, expert_dim):
    return int(tokens_per_device * capacity_factor // expert_dim)


class SwitchRouter(nn.Cell):
    """SwitchRouter"""

    def __init__(self,
                 d_model,
                 expert_num,
                 capacity_factor,
                 config,
                 is_training=True):
        super(SwitchRouter, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.expert_dim = expert_num
        self.d_model = d_model
        self.input_shape = (-1, self.d_model)
        self.expert_parallel = config.ep
        self.capacity_factor = capacity_factor
        self.matmul = P.MatMul().shard(((config.dp, 1), (1, 1)))
        self.weight_shape = (self.d_model, self.expert_dim)
        self.router_weight = Parameter(initializer('normal', self.weight_shape, mstype.float32), name="wg",
                                       parallel_optimizer=False)

        self.is_training = is_training
        self.cast = P.Cast()
        self.softmax = P.Softmax(axis=-1).shard(((config.dp, 1, 1),))
        self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False).shard(((config.dp, 1, 1),))
        self.onehot = P.OneHot().shard(((config.dp, 1, 1), (), ()))
        self.onehot2 = P.OneHot().shard(((config.dp, 1, 1), (), ()))
        self.onehot3 = P.OneHot().shard(((config.dp, 1, 1, 1), (), ()))
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((config.dp, 1, 1),))
        self.reduce_mean2 = P.ReduceMean(keep_dims=False).shard(((config.dp, 1, 1),))
        self.reduce_mean3 = P.ReduceMean(keep_dims=False).shard(((config.dp, 1),))
        self.mul = P.Mul().shard(((config.dp, 1), (config.dp, 1)))
        self.mul2 = P.Mul().shard(((), ()))
        self.mul3 = P.Mul().shard(((), ()))
        self.mul4 = P.Mul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.mul5 = P.Mul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.mul6 = P.Mul().shard(((config.dp, 1), (config.dp, 1)))
        self.mul7 = P.Mul().shard(((config.dp, 1), (config.dp, 1)))
        self.mul8 = P.Mul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.mul9 = P.Mul().shard(((config.dp, 1, 1, 1), (config.dp, 1, 1, 1)))

        self.cumsum = CumSum(config=config)
        self.less = P.Less().shard(((config.dp, 1, 1), ()))
        self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((config.dp, 1, 1),))
        self.expand = P.ExpandDims().shard(((config.dp, 1),))
        self.expand2 = P.ExpandDims().shard(((config.dp, 1, 1),))
        self.pri = P.Print()
        self.not_equal = P.NotEqual().shard(((config.dp, 1, 1, 1), ()))

    def construct(self, input_tensor):
        """Construct method"""
        input_tensor = self.reshape(input_tensor, self.input_shape)
        router_weight = self.cast(self.router_weight, mstype.float16)
        router_logits = self.matmul(input_tensor, router_weight)
        bs_and_expert_dim = self.shape(router_logits)

        router_logits = self.cast(router_logits, mstype.float32)
        tokens_per_device = bs_and_expert_dim[0] / self.expert_parallel
        expert_capacity = caculate_expert_capacity(tokens_per_device, self.capacity_factor, self.expert_dim)
        router_logits = self.reshape(router_logits, (self.expert_parallel, tokens_per_device, self.expert_dim))

        router_prob = self.softmax(router_logits)
        expert_index, expert_gate = self.argmax(router_prob)
        expert_mask = self.onehot(expert_index, self.expert_dim, self.on_value, self.off_value)

        density_1 = self.reduce_mean(expert_mask, 1)
        density_1_proxy = self.reduce_mean2(router_prob, 1)
        loss = self.mul(density_1, density_1_proxy)
        loss = self.reduce_mean3(loss)
        loss = self.mul3(self.mul2(loss, self.expert_dim), self.expert_dim)

        cumsum = self.cumsum(expert_mask, tokens_per_device)
        position_in_expert = self.mul4(cumsum, expert_mask)
        less_result = self.less(position_in_expert, expert_capacity)
        expert_mask = self.mul5(less_result, expert_mask)
        expert_mask_flat = self.reduce_sum(expert_mask, -1)

        expert_gate = self.mul6(expert_gate, expert_mask_flat)

        combine_tensor = self.mul7(expert_gate, expert_mask_flat)
        combine_tensor = self.mul8(self.expand(combine_tensor, -1),
                                   self.onehot2(expert_index, self.expert_dim, self.on_value, self.off_value))
        combine_tensor = self.mul9(self.expand2(combine_tensor, -1),
                                   self.onehot3(self.cast(position_in_expert, mstype.int32), expert_capacity,
                                                self.on_value, self.off_value))
        combine_tensor = self.cast(combine_tensor, mstype.float16)

        # dispatch_tensor = self.cast(combine_tensor, mstype.bool_)
        dispatch_tensor = self.not_equal(combine_tensor, 0)
        return dispatch_tensor, combine_tensor, loss


class MoE(nn.Cell):
    """MoE"""

    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 config,
                 hidden_act="relu",
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=mstype.float32,
                 init_method_std=0.02,
                 num_layers=6,
                 layer_index=0):
        super(MoE, self).__init__()
        self.expert_parallel = config.ep
        self.capacity_factor = config.capacity_factor
        self.expert_dim = config.expert_num
        self.aux_loss_factor = config.aux_loss_factor
        self.d_model = in_channels
        self.hidden_size = hidden_size
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        self.router = SwitchRouter(d_model=self.d_model,
                                   expert_num=self.expert_dim,
                                   capacity_factor=self.capacity_factor,
                                   config=config,
                                   is_training=True)

        self.wi_shape = (self.expert_dim, self.d_model, self.hidden_size)
        self.wi = Parameter(initializer('normal', self.wi_shape, mstype.float32), name="wi")
        self.wo_shape = (self.expert_dim, self.hidden_size, self.d_model)
        self.wo = Parameter(initializer('normal', self.wo_shape, mstype.float32), name="wo")
        self.transpose = P.Transpose().shard(((config.dp, 1, 1),))
        self.transpose2 = P.Transpose().shard(((config.dp, 1, 1, 1),))
        self.transpose3 = P.Transpose().shard(((config.ep, 1, 1, 1),))
        self.transpose4 = P.Transpose().shard(((config.dp, 1, 1),))
        self.transpose5 = P.Transpose().shard(((config.dp, 1, 1),))
        self.transpose_two = P.Transpose().shard(((config.ep, 1),))

        self.batch_mm = P.BatchMatMul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.batch_mm2 = P.BatchMatMul().shard(((config.ep, 1, 1), (config.ep, 1, config.mp)))
        self.batch_mm3 = P.BatchMatMul().shard(((config.ep, 1, config.mp), (config.ep, config.mp, 1)))
        self.batch_mm4 = P.BatchMatMul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.gelu = P.GeLU().shard(((config.ep, 1, config.mp),))
        self.preprocess = LayerNorm2((in_channels,), config.dp).to_float(mstype.float32)
        self.dropout = Dropout(1 - hidden_dropout_prob)
        self.dropout.shard(((config.dp, 1),))
        self.add = P.Add().shard(((config.dp, 1), (config.dp, 1)))
        self.mul = P.Mul().shard(((), ()))

    def construct(self, input_tensor):
        """Construct method"""
        origin_shape = input_tensor.shape
        input_tensor = input_tensor.view(-1, self.d_model)
        inputs = self.preprocess(input_tensor)
        inputs = inputs.astype(mstype.float16)
        bs_and_model = self.shape(inputs)
        tokens_per_device = bs_and_model[0] / self.expert_parallel
        inputs = inputs.view(self.expert_parallel, tokens_per_device, self.d_model)
        expert_capacity = caculate_expert_capacity(tokens_per_device, self.capacity_factor, self.expert_dim)
        dispatch_tensor, combine_tensor, aux_loss = self.router(inputs)
        inputs = self.transpose(inputs, (0, 2, 1))
        dispatch_tensor = dispatch_tensor.view(self.expert_parallel, tokens_per_device,
                                               self.expert_dim * expert_capacity)
        dispatch_tensor = dispatch_tensor.astype(mstype.float16)
        expert_input = self.batch_mm(inputs, dispatch_tensor)
        expert_input = expert_input.view(self.expert_parallel, self.d_model, self.expert_dim, expert_capacity)

        shape = expert_input.shape
        expert_input = self.transpose2(expert_input, (0, 3, 1, 2))
        expert_input = expert_input.view(-1, shape[2])
        expert_input = self.transpose_two(expert_input, (1, 0))
        expert_input = expert_input.view(shape[2], shape[0], shape[3], shape[1])
        # expert_input = self.transpose2(expert_input, (2, 0, 3, 1))

        expert_input = expert_input.view(self.expert_dim, self.expert_parallel * expert_capacity, self.d_model)
        wi = self.wi.astype(mstype.float16)
        output = self.batch_mm2(expert_input, wi)
        output = self.gelu(output)
        wo = self.wo.astype(mstype.float16)
        output = self.batch_mm3(output, wo)
        output = output.view(self.expert_dim, self.expert_parallel, expert_capacity, self.d_model)

        shape1 = output.shape
        output = self.transpose2(output, (0, 2, 1, 3))
        output = output.view(-1, shape1[1] * shape1[3])
        output = self.transpose_two(output, (1, 0))
        output = output.view(shape1[1], shape1[3], shape1[0], shape1[2])
        # output = self.transpose3(output, (1, 3, 0, 2))

        output = output.view(self.expert_parallel, self.d_model, self.expert_dim * expert_capacity)
        combine_tensor = combine_tensor.view(self.expert_parallel, tokens_per_device, self.expert_dim * expert_capacity)
        combine_tensor = self.transpose4(combine_tensor, (0, 2, 1))
        combined_output = self.batch_mm4(output, combine_tensor)
        combined_output = self.transpose5(combined_output, (0, 2, 1))
        combined_output = combined_output.view(bs_and_model)
        output = self.dropout(combined_output)
        output = self.add(output, input_tensor)
        output = output.view(origin_shape)
        aux_loss = self.mul(self.aux_loss_factor, aux_loss)
        return output, aux_loss
