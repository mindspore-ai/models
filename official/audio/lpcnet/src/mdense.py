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
from mindspore import Parameter, nn
from mindspore.common.initializer import initializer


class MDense(nn.Cell):
    """ Implementation of dual fully-connected layer """
    def __init__(self,
                 input_dim,
                 outputs,
                 channels=2,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='xavier_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.outputs = outputs
        self.activation = activation() if activation else None

        self.dense_layers = nn.CellList([nn.Dense(input_dim, outputs, kernel_initializer, bias_initializer,
                                                  use_bias, activation='tanh') for _ in range(self.channels)])

        self.factor = mindspore.ParameterTuple([Parameter(initializer('ones', (outputs,)), name=f"factor_{i}") \
                                                for i in range(self.channels)])

    def construct(self, inputs):
        output = self.dense_layers[0](inputs) * self.factor[0]
        for i in range(1, self.channels):
            output += self.dense_layers[i](inputs) * self.factor[i]

        if self.activation is not None:
            # pylint: disable=not-callable
            output = self.activation(output)
        return output
