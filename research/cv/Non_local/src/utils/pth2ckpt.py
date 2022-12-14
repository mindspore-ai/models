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
"""
python pth2ckpt.py
"""
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

param_info = {
    'bn1.weight': 'bn1.bn2d.gamma',
    'bn1.bias': 'bn1.bn2d.beta',
    'bn1.running_mean': 'bn1.bn2d.moving_mean',
    'bn1.running_var': 'bn1.bn2d.moving_variance',
    'bn2.weight': 'bn2.bn2d.gamma',
    'bn2.bias': 'bn2.bn2d.beta',
    'bn2.running_mean': 'bn2.bn2d.moving_mean',
    'bn2.running_var': 'bn2.bn2d.moving_variance',
    'bn3.weight': 'bn3.bn2d.gamma',
    'bn3.bias': 'bn3.bn2d.beta',
    'bn3.running_mean': 'bn3.bn2d.moving_mean',
    'bn3.running_var': 'bn3.bn2d.moving_variance',
    'downsample.1.running_mean': 'downsample.1.bn2d.moving_mean',
    'downsample.1.running_var': 'downsample.1.bn2d.moving_variance',
    'downsample.1.weight': 'downsample.1.bn2d.gamma',
    'downsample.1.bias': 'downsample.1.bn2d.beta'
}

def pytorch2mindspore():
    """

    Returns:
        object:
    """
    par_dict = torch.load('resnet50-19c8e357.pth', map_location='cpu')
    params = list(par_dict.keys())
    new_params_list = []
    for param in params:
        param_dict = {}
        data = par_dict[param]
        param_names = param_info.keys()
        for key in param_names:
            if key in param:
                param = param.replace(key, param_info[key])
            param_dict['name'] = param
            param_dict['data'] = Tensor(data.detach().numpy())
            new_params_list.append(param_dict)
    save_checkpoint(new_params_list, 'resnet50.ckpt')

pytorch2mindspore()
