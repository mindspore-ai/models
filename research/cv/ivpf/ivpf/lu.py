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
"""
Layers with weights LU decomposition.
"""
import numpy as np
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype

import mindspore.nn as nn
from mindspore.ops import operations as ops
import mindspore.numpy as mnp

from .roundquant import RoundQuant


class LULinear(nn.Cell):
    """Linear layer with LU decomposed weights."""
    def __init__(self, features, args, identity_init=True):
        super(LULinear, self).__init__()

        self.features = features
        self.identity_init = identity_init

        self.tril_mask = Tensor(np.tril(np.ones((features, features)), k=-1), mstype.float32)
        self.triu_mask = Tensor(np.triu(np.ones((features, features)), k=1), mstype.float32)
        self.weights = Parameter(
            Tensor(
                np.random.randn(
                    features,
                    features) /
                features,
                mstype.float32),
            name='w',
            requires_grad=True)
        self.bias = Parameter(mnp.zeros(features, mstype.float32), name='b', requires_grad=True)

        self._initialize(identity_init)

        if args.variable_type == 'discrete':
            self.rnd = RoundQuant(2**args.n_bits)
            self.set_grad(False)
        else:
            self.rnd = None

        self.matmul = ops.MatMul(transpose_b=True)
        self.zeros_like = ops.ZerosLike()

    def _initialize(self, identity_init):
        pass

    def _create_lower_upper(self):
        """get lower and upper traingular matrix of weights"""
        lower = self.tril_mask * self.weights
        upper = self.triu_mask * self.weights

        return lower, upper

    def construct(self, inputs, ldj, reverse=False):
        """construct"""
        lower, upper = self._create_lower_upper()
        if self.rnd is None:
            if not reverse:
                outputs = self.matmul(inputs, upper) + inputs
                outputs = self.matmul(outputs, lower) + outputs
                outputs = outputs + self.bias
            else:
                outputs = inputs - self.bias

                for i in range(1, outputs.shape[1]):
                    outputs[:, i:i + 1] -= self.matmul(outputs[:, :i], lower[i:i + 1, :i])

                for i in range(outputs.shape[1] - 2, -1, -1):
                    outputs[:, i:i + 1] -= self.matmul(outputs[:, i + 1:], upper[i:i + 1, i + 1:])
        else:
            inputs = mnp.reshape(mnp.ravel(inputs), inputs.shape)
            if not reverse:
                outputs = mnp.concatenate([self.matmul(inputs[:, i + 1:], upper[i:i + 1, i + 1:])
                                           for i in range(inputs.shape[1] - 1)], axis=1)
                outputs = mnp.concatenate([outputs, self.zeros_like(inputs[:, -1:])], axis=1)
                outputs = self.rnd(outputs)
                outputs += inputs

                out1 = mnp.concatenate([self.matmul(outputs[:, :i], lower[i:i + 1, :i])
                                        for i in range(1, inputs.shape[1])], axis=1)
                out1 = mnp.concatenate([self.zeros_like(inputs[:, -1:]), out1], axis=1)
                out1 = self.rnd(out1)
                out1 += outputs

                outputs = out1 + self.rnd(self.bias)
            else:
                outputs = inputs - self.rnd(self.bias)

                for i in range(1, outputs.shape[1]):
                    outputs[:, i:i + 1] -= self.rnd(self.matmul(outputs[:, :i], lower[i:i + 1, :i]))

                for i in range(outputs.shape[1] - 2, -1, -1):
                    outputs[:, i:i + 1] -= self.rnd(self.matmul(outputs[:, i + 1:], upper[i:i + 1, i + 1:]))

        return outputs, ldj


class LUConv1x1(LULinear):
    """1x1 convolution layer with LU decomposed weights."""
    def __init__(self, num_channels, args, identity_init=True):
        super(LUConv1x1, self).__init__(num_channels, args, identity_init)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def _lu_forward_inverse(self, inputs, ldj, reverse=False):
        """convert 1x1 convolution to linear transform"""
        b, c, h, w = inputs.shape
        inputs = self.transpose(inputs, (0, 2, 3, 1))
        inputs = self.reshape(inputs, (b * h * w, c))

        outputs, ldj = super(LUConv1x1, self).construct(inputs, ldj, reverse)

        outputs = self.reshape(outputs, (b, h, w, c))
        outputs = self.transpose(outputs, (0, 3, 1, 2))

        return outputs, ldj

    def construct(self, inputs, ldj, reverse=False):
        """construct"""
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        outputs, ldj = self._lu_forward_inverse(inputs, ldj, reverse)

        return outputs, ldj
