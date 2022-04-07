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
"""Sub-module of MMoE model"""
import mindspore.nn as nn
import mindspore as ms
from mindspore import Parameter
from mindspore.common.initializer import initializer, TruncatedNormal, Zero
import mindspore.ops as P

from src.model_utils.config import config

if config.device_target == 'Ascend':
    use_mstype = ms.float16
else:
    use_mstype = ms.float32


class expert(nn.Cell):
    """expert network"""

    def __init__(self,
                 input_size,
                 units,
                 num_experts,
                 use_expert_bias=True,
                 ):
        super(expert, self).__init__()
        self.input_size = int(input_size)
        self.units = int(units)
        self.num_experts = int(num_experts)
        self.use_expert_bias = use_expert_bias
        self.mul = P.tensor_dot
        self.bias_add = P.Add()
        self.relu = nn.ReLU()
        self.expert_kernels = Parameter(initializer(TruncatedNormal(),
                                                    (self.input_size, self.units,
                                                     self.num_experts),
                                                    ms.float32),
                                        requires_grad=True)
        if self.use_expert_bias:
            self.expert_bias = Parameter(initializer(Zero(),
                                                     (self.units, self.num_experts),
                                                     ms.float32),
                                         requires_grad=True)
        else:
            self.expert_bias = None
        self.print = P.Print

    def construct(self, x):
        """construct of expert network"""
        # expert_output = self.mul(x.astype(ms.float16), self.expert_kernels.astype(ms.float16), (1, 0))
        expert_output = self.mul(
            x, self.expert_kernels.astype(use_mstype), (1, 0))
        if self.use_expert_bias:
            expert_output = self.bias_add(
                expert_output, self.expert_bias.astype(use_mstype))
        expert_output = self.relu(expert_output)
        return expert_output


class gate(nn.Cell):
    """gate network"""

    def __init__(self,
                 input_size,
                 num_experts,
                 use_gate_bias=True,
                 ):
        super(gate, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.use_gate_bias = use_gate_bias
        self.bias_add = P.BiasAdd()
        self.softmax = nn.Softmax()
        self.gate_kernel = Parameter(initializer(TruncatedNormal(),
                                                 (self.input_size,
                                                  self.num_experts),
                                                 ms.float32),
                                     requires_grad=True)

        if self.use_gate_bias:
            self.gate_bias = Parameter(initializer(Zero(),
                                                   (self.num_experts,),
                                                   ms.float32),
                                       requires_grad=True)
        else:
            self.gate_bias = None
        self.print = P.Print()

    def construct(self, x):
        """construct of gate network"""
        # gate_output = P.dot(x1=x.astype(ms.float16), x2=self.gate_kernel.astype(ms.float16))
        gate_output = P.dot(x1=x, x2=self.gate_kernel.astype(use_mstype))
        if self.use_gate_bias:
            gate_output = self.bias_add(
                gate_output, self.gate_bias.astype(use_mstype))
        gate_output = self.softmax(gate_output)
        return gate_output


class shared_output(nn.Cell):
    """Gate controls the weights of different experts for different tasks"""

    def __init__(self,
                 input_size,
                 num_experts,
                 units):
        super(shared_output, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.units = units
        self.expand_dims = P.ExpandDims()
        self.repeat_elements = P.repeat_elements
        self.sum = P.ReduceSum()
        self.print = P.Print()

    def construct(self, x, x1):
        """construct of shared output"""
        expanded_gate_output = self.expand_dims(x1, 1)
        weighted_expert_output = x * \
            self.repeat_elements(x=expanded_gate_output,
                                 rep=self.units, axis=1)
        final_outputs = self.sum(weighted_expert_output, 2)

        return final_outputs


class tower(nn.Cell):
    """dense with TRelu activation"""

    def __init__(self, in_channels, out_channels):
        super(tower, self).__init__()
        self.relu = nn.ReLU()
        self.tower_layer = nn.Dense(in_channels,
                                    out_channels,
                                    weight_init=TruncatedNormal(),
                                    activation=self.relu)
        self.tower_layer.to_float(use_mstype)
        self.print = P.Print()

    def construct(self, x):
        """construct of tower layer"""
        x = self.tower_layer(x)
        return x


class output(nn.Cell):
    """dense with TSoftmax activation"""

    def __init__(self, in_channels, out_channels):
        super(output, self).__init__()
        self.softmax = nn.Softmax()
        self.output = nn.Dense(in_channels,
                               out_channels,
                               weight_init=TruncatedNormal(),
                               activation=self.softmax)
        self.output.to_float(use_mstype)
        self.print = P.Print()

    def construct(self, x):
        """construct of output layer"""
        x = self.output(x)
        return x
