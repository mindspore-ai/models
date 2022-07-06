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
"""Util class or function."""


def get_param_groups(trainable_params):
    """Param groups for optimizer."""

    first_conv_weight = []
    first_conv_bias = []
    first_bn = []

    normal_weight = []
    normal_bias = []

    for x in trainable_params:

        # First layer
        if x.name.endswith('conv1_7x7_s2.conv.weight'):
            first_conv_weight.append(x)
        elif x.name.endswith('conv1_7x7_s2.conv.bias'):
            first_conv_bias.append(x)
        elif x.name.endswith('conv1_7x7_s2.bn.gamma'):
            first_bn.append(x)
        elif x.name.endswith('conv1_7x7_s2.bn.beta'):
            first_bn.append(x)

        # Other layers (conv and linear)
        elif x.name.endswith('.weight'):
            normal_weight.append(x)
        elif x.name.endswith('.bias'):
            normal_bias.append(x)
        else:
            raise ValueError(f"Unknown parameter: {x.name}")

    return first_conv_weight, first_conv_bias, first_bn, normal_weight, normal_bias
