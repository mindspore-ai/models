# Copyright 2023 Huawei Technologies Co., Ltd
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
network
"""
import mindspore.nn as nn


class MsMLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(MsMLP, self).__init__()
        net_layer = [nn.Dense(input_dim, hidden_dim, has_bias=True, weight_init='Uniform', bias_init='Uniform'),
                     nn.Tanh()]
        for i in range(n_layers):
            if i != n_layers - 1:
                net_layer.append(nn.Dense(hidden_dim,
                                          hidden_dim,
                                          has_bias=True,
                                          weight_init='Uniform',
                                          bias_init='Uniform'))
                net_layer.append(nn.Tanh())
            else:
                net_layer.append(
                    nn.Dense(hidden_dim, output_dim, has_bias=True, weight_init='Uniform', bias_init='Uniform'))
        self.seq = nn.SequentialCell(net_layer)

    def construct(self, x):
        return self.seq(x)
