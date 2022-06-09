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
"""convert h5 weight to ckpt"""
from mindspore import Tensor, Parameter

import h5py
import numpy as np

def translate_h5(h5path=''):
    """convert h5 weight to ckpt"""
    count = 0
    f = h5py.File(h5path, 'r')
    weights = {}
    for i in list(f.keys()):
        for name in list(f[i]):
            if name.startswith('conv1') or name.startswith('bn_conv1'):
                continue
            param_name = name.replace('_W_1:0', '.weight')
            param_name = param_name.replace('_b_1:0', '.bias')
            param_name = param_name.replace('_gamma_1:0', '.gamma')
            param_name = param_name.replace('_beta_1:0', '.beta')
            param_name = param_name.replace('_running_mean_1:0', '.moving_mean')
            param_name = param_name.replace('_running_std_1:0', '.moving_variance')
            data = f[i][name][:].astype(np.float32)
            count += 1
            weights[param_name] = data
    parameter_dict = {}
    for name in weights:
        tensor_data = Tensor.from_numpy(weights[name])
        if len(weights[name].shape) == 4:
            tensor_data = tensor_data.transpose(3, 2, 1, 0)
        if len(weights[name].shape) == 2:
            tensor_data = tensor_data.transpose(1, 0)
        parameter_dict[name] = Parameter(tensor_data, name=name)
    return parameter_dict
