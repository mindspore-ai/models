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

import mindspore
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import constexpr
import mindspore.numpy as np
from mindspore.common.initializer import initializer, XavierUniform
from src import utils


@constexpr
def get_key(key):
    return str(key)


def get_weights(shape):
    nn_param = initializer(XavierUniform(), shape, mindspore.float32)
    nn_param = mindspore.Parameter(nn_param, name='weight_'+str(shape))
    return nn_param

def get_biases(param):
    length, bias_start = param
    biases = initializer(bias_start, length, mindspore.float32)
    biases = mindspore.Parameter(biases, name='bias_'+str(length))
    return biases

class DCGRUCell(mindspore.nn.Cell):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, is_fp16, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        super().__init__()
        self._activation = mindspore.ops.Tanh() if nonlinearity == 'tanh' else mindspore.ops.ReLU()
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.is_fp16 = is_fp16
        supports = []

        self.cast = P.Cast()
        self.sigmoid = ops.Sigmoid()
        self.reshape = ops.Reshape()
        self.split = ops.Split(-1, 2)
        self.concat = ops.Concat()
        self.concat_two = ops.Concat(2)
        self.transpose = ops.Transpose()
        self.zeros = ops.Zeros()
        self.matmul = ops.MatMul()
        self.expand_dims = ops.ExpandDims()
        self._weights = {}
        self._biases = {}
        shape = [(330, 128), (330, 64), (325, 128), (325, 64), (640, 128), (640, 64)]
        for i in shape:
            self._weights[str(i)] = get_weights(i)
        length = [(64, 0.0), (128, 1.0)]
        for j in length:
            self._biases[str(j)] = get_biases(j)

        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))


    def _convert(self, T, data, shape):
        L = self.zeros((shape[0], shape[1]), mindspore.float32)
        columns = T.shape[1]
        for i in range(columns):
            x = T[0][i]
            y = T[1][i]
            L[x][y] = self.cast(data[i], mstype.float32)
        return L

    def _build_sparse_matrix(self, L):
        x1 = Tensor(L.row, mstype.int32)
        x2 = Tensor(L.col, mstype.int32)
        indices = np.column_stack((x1, x2))
        L = self._convert(indices.T, L.data, L.shape)
        return L

    def construct(self, inputs, hx):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = self.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = self.reshape(value, (-1, self._num_nodes, output_size))
        r, u = self.split(value)
        r = self.reshape(r, (-1, self._num_nodes * self._num_units))
        u = self.reshape(u, (-1, self._num_nodes * self._num_units))
        c = self._gconv(inputs, r * hx, self._num_units)

        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    def _concat(self, x, x_):
        x_ = self.expand_dims(x_, 0)
        return self.concat((x, x_))

    def _fc(self, inputs, state, output_size, bias_start=0.0):

        batch_size = inputs.shape[0]
        inputs = self.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = self.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = self.concat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]

        weights = self._weights[get_key((input_size, output_size))]
        biases = self._biases[get_key((output_size, bias_start))]

        value = self.sigmoid(self.matmul(inputs_and_state, weights))
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = self.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = self.reshape(state, (batch_size, self._num_nodes, -1))
        if self.is_fp16:
            state = self.cast(state, mstype.float16)  # amp

        inputs_and_state = self.concat_two([inputs, state])
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = self.transpose(x, (1, 2, 0))
        x0 = self.reshape(x0, (self._num_nodes, input_size * batch_size))
        x = self.expand_dims(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                if self.is_fp16:
                    support = self.cast(support, mstype.float16)  # amp

                x1 = self.matmul(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    k = k
                    x2 = 2 * self.matmul(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = self.reshape(x, (num_matrices, self._num_nodes, input_size, batch_size))
        x = self.transpose(x, (3, 1, 2, 0))
        x = self.reshape(x, (batch_size * self._num_nodes, input_size * num_matrices))

        weights = self._weights[get_key((input_size * num_matrices, output_size))]
        biases = self._biases[get_key((output_size, bias_start))]

        if self.is_fp16:
            weights = self.cast(weights, mstype.float16)  # amp

        x = self.matmul(x, weights)
        x += biases
        return self.reshape(x, (batch_size, self._num_nodes * output_size))
