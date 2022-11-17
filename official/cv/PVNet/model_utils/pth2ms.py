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
"""pytorh to mindspore functions"""
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import torch


def pth2ms(pth_file_path, net):
    """pytorh to mindspore"""
    par_dict = torch.load(pth_file_path, map_location='cpu')['net']
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        ndim = parameter.numpy().ndim
        print('before:', name, parameter.shape)
        if ndim < 2 and not name.startswith('convraw.3'):
            if name.endswith('bias'):
                name = name.replace('bias', 'beta')
            elif name.endswith('weight'):
                name = name.replace('weight', 'gamma')
            elif name.endswith('running_mean'):
                name = name.replace('running_mean', 'moving_mean')
            elif name.endswith('running_var'):
                name = name.replace('running_var', 'moving_variance')
        if net == 'resnet18':
            name = 'resnet18_8s.' + name
        print('after:', name, parameter.shape)
        if ndim > 0:
            param_dict['name'] = name
            param_dict['data'] = Tensor(parameter.numpy())
            params_list.append(param_dict)
    return params_list


def resnet_pth2ms(pth_file_path):
    """reset pytorh to mindspore"""
    par_dict = torch.load(pth_file_path, map_location='cpu')
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        ndim = parameter.detach().numpy().ndim
        if ndim < 2 and not name.startswith('fc'):
            if name.endswith('bias'):
                name = name.replace('bias', 'beta')
            elif name.endswith('weight'):
                name = name.replace('weight', 'gamma')
            elif name.endswith('running_mean'):
                name = name.replace('running_mean', 'moving_mean')
            elif name.endswith('running_var'):
                name = name.replace('running_var', 'moving_variance')

        print('after:', name, parameter.shape)
        if ndim > 0:
            param_dict['name'] = name
            param_dict['data'] = Tensor(parameter.detach().numpy())
            params_list.append(param_dict)
    return params_list


if __name__ == "__main__":
    pth_name = "resnet18-5c106cde.pth"
    new_params_list = resnet_pth2ms(pth_name)
    new_ckpt_path = "resnet18-5c106cde.ckpt"
    save_checkpoint(new_params_list, new_ckpt_path)
