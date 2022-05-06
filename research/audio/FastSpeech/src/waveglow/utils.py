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
"""Utils scripts."""
from mindspore import ops


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Fusion method.
    """
    n_channels_int = n_channels
    in_act = input_a + input_b

    t_act = ops.Tanh()(in_act[:, :n_channels_int, :])
    s_act = ops.Sigmoid()(in_act[:, n_channels_int:, :])

    acts = t_act * s_act

    return acts


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames.
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]

    return files
