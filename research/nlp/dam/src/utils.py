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
"""Operation Utils"""
import numpy as np
from mindspore import nn
from mindspore import ops as P
from mindspore import Tensor
from mindspore import numpy as mnp
from mindspore import Parameter
from mindspore.common import dtype as mstype
from mindspore.common import initializer


class DotSim(nn.Cell):
    '''calculate dot similarity with two tensor.

    Args:
        x: a tensor with shape [batch, time_x, dimension]
        y: a tensor with shape [batch, time_y, dimension]

    Returns:
        a tensor with shape [batch, time_x, time_y]
    Raises:
        AssertionError: if
            the shapes of x and y are not match.
    '''

    def __init__(self, is_norm=True):
        super(DotSim, self).__init__()
        self.is_norm = is_norm
        self.sqrt = P.Sqrt()
        self.maximum = P.Maximum()
        self.cast = P.Cast()
        self.batch_matmul_trans_b = BatchMatMulCell(transpose_a=False, transpose_b=True)

    def construct(self, x, y):
        sim = self.batch_matmul_trans_b(x, y)
        if self.is_norm:
            scale = self.sqrt(self.cast(x.shape[-1], mstype.float32))
            scale = self.maximum(scale, 1.0)
            sim = sim / scale
        return sim


def orthogonal(shape):
    """Generating orthogonal matrix"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def orthogonal_init(shape, gain=1.0):
    """Generating orthogonal matrix"""
    # Check the shape
    if len(shape) < 2:
        raise ValueError("The tensor to initialize must be "
                         "at least two-dimensional")
    # Flatten the input shape with the last dimension remaining
    # its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)

    # Generate a random matrix
    a = np.random.normal(size=flat_shape).astype(np.float32)
    # Compute the qr factorization
    q, r = np.linalg.qr(a, mode='reduced')
    # Make Q uniform
    square_len = np.minimum(num_rows, num_cols)
    d = np.diagonal(r[:square_len, :square_len])
    ph = d / np.absolute(d)
    q *= ph
    # Pad zeros to Q (if rows smaller than cols)
    if num_rows < num_cols:
        padding = np.zeros([num_rows, num_cols - num_rows], dtype=np.float32)
        q = np.concatenate([q, padding], 1)
    return gain * np.reshape(q, shape)


class BilinearSim(nn.Cell):
    """Calculate bilinear similarity with two tensor."""
    def __init__(self, is_norm=True):
        super(BilinearSim, self).__init__()
        self.is_norm = is_norm
        self.cast = P.Cast()
        self.transpose = P.Transpose()

    def construct(self, x, y):
        """Calculate bilinear similarity with two tensor (x, y)."""
        M = Tensor(orthogonal_init([x.shape[-1], y.shape[-1]]),
                   dtype=mstype.float32)
        y = self.transpose(y, (1, 0))
        sim = P.matmul(P.matmul(x, M), y)
        if self.is_norm:
            scale = P.sqrt(self.cast(x.shape[-1] * y.shape[-1], mstype.float32))
            scale = P.maximum(1.0, scale)
            sim = sim / scale
        return sim


class PositionalEncodingVector(nn.Cell):
    """Adds a bunch of sinusoids of different frequencies to a tensor."""
    def __init__(self, length, channels, min_timescale=1.0, max_timescale=1.0e4, value=0):
        super(PositionalEncodingVector, self).__init__()
        self.length = length
        self.channels = channels
        self.mul = P.Mul()
        self.min_timescale = Tensor(min_timescale, mstype.float32)
        self.max_timescale = Tensor(max_timescale, mstype.float32)
        self.num_timescales = Tensor(self.channels // 2, mstype.float32)
        self.value = value
        self._lambda = Parameter(initializer.initializer(self.value, [self.length, 1], mstype.float32), name='lambda')

        self.cast = P.Cast()
        self.exp = P.Exp()
        self.expand_dims = P.ExpandDims()
        self.sin = P.Sin()
        self.cos = P.Cos()
        self.concat = P.Concat(axis=1)
        self.mod = P.Mod()
        self.pad = P.Pad(((0, 0), (0, self.channels % 2)))

    def construct(self, x):
        """Generate position code"""
        position = nn.Range(self.length)()
        log_timescale_increment = (P.log(self.max_timescale / self.min_timescale) / (self.num_timescales - 1))
        inv_timescales = self.min_timescale * self.exp(
            nn.Range(self.num_timescales.data)() * -log_timescale_increment)
        scaled_time = self.expand_dims(position, 1) * self.expand_dims(inv_timescales, 0)
        signal = self.concat([self.sin(scaled_time), self.cos(scaled_time)])
        signal = self.pad(signal)
        signal = self.mul(self._lambda, signal)
        signal = self.expand_dims(signal, 0)

        return x + signal


class BatchMatMulCell(nn.Cell):
    """BatchMatMulCell compute with fp16"""
    def __init__(self, transpose_a=False, transpose_b=False):
        super(BatchMatMulCell, self).__init__()
        self.batch_matmul = P.BatchMatMul(transpose_a=transpose_a, transpose_b=transpose_b)
        self.cast = P.Cast()

    def construct(self, x, y):
        """Convert the input to fp16 for calculation"""
        x = self.cast(x, mstype.float16)
        y = self.cast(y, mstype.float16)
        out = self.batch_matmul(x, y)
        out = self.cast(out, mstype.float32)
        return out


def get_mask(row_lengths, col_lengths, max_row_length, max_col_length):
    '''Return a mask tensor representing the first N positions of each row and each column.

    Args:
        row_lengths: a tensor with shape [batch]
        col_lengths: a tensor with shape [batch]

    Returns:
        a mask tensor with shape [batch, max_row_length, max_col_length]

    '''
    row_mask = get_sequence_mask(row_lengths, max_row_length)  # bool, [batch, max_row_len]
    row_mask = P.Reshape()(row_mask, (-1, max_row_length))
    col_mask = get_sequence_mask(col_lengths, max_col_length)  # bool, [batch, max_col_len]
    col_mask = P.Reshape()(col_mask, (-1, max_col_length))

    row_mask = P.cast(P.expand_dims(row_mask, -1), mstype.float32)
    col_mask = P.cast(P.expand_dims(col_mask, -1), mstype.float32)

    mask = BatchMatMulCell(transpose_b=True)(row_mask, col_mask)
    return mask


def get_sequence_mask(lengths, maxlen=None):
    """
    Returns a mask tensor representing the first N positions of each cell.

    If lengths has shape [d_1, d_2, ..., d_n], then the resulting tensor mask has type dtype and shape
    [d_1, d_2, ..., d_n, maxlen], with mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])

    Inputs:
        - **lengths** (Tensor) - Tensor to calculate the mask for. All values in this tensor should be
          less than or equal to `maxlen`. Values greater than `maxlen` will be treated as `maxlen`.
          Must be type int32 or int64.

        - **maxlen** (int) - size of the last dimension of returned tensor. Must be positive and same
          type as elements in `lengths`.

    Outputs:
        One mask tensor of shape lengths.shape + (maxlen,).

    """
    argmax_op = P.ArgMaxWithValue()
    reshape_op = P.Reshape()
    cast_op = P.Cast()

    if maxlen is None:
        flatten_data = reshape_op(lengths, (-1,))
        flatten_data = cast_op(flatten_data, mstype.float32)
        _, value = argmax_op(flatten_data)
        maxlen = value.asnumpy()
    else:
        maxlen = maxlen

    range_vector = mnp.arange(0, maxlen, 1, dtype=mstype.int32)
    mask = P.ExpandDims()(lengths, -1)
    result = range_vector < mask
    return result
