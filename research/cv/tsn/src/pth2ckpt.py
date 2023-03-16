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
generate ckpt from pth.
"""
import argparse
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

parser = argparse.ArgumentParser(description="convert torch pretrain model to mindspore checkpoint.")
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--modality', type=str, default="RGB", choices=['RGB', 'Flow'])
parser.add_argument('--new_length', type=int, default=5)

modules = {'inception_3a', 'inception_3b', 'inception_3c',
           'inception_4a', 'inception_4b', 'inception_4c',
           'inception_4d', 'inception_4e', 'inception_5a', 'inception_5b'}


def checkname(name):
    if 'bn' in name:
        if name.endswith('bias'):
            name = name.replace('bias', 'beta')
        if name.endswith('weight'):
            name = name.replace('weight', 'gamma')
        if name.endswith('running_mean'):
            name = name.replace('running_mean', 'moving_mean')
        if name.endswith('running_var'):
            name = name.replace('running_var', 'moving_variance')
    if 'bn' not in name and 'fc' not in name and '7x7' not in name:
        if 'bias' in name:
            name = name.replace('bias', 'conv.bias')
        else:
            name = name.replace('weight', 'conv.weight')
    if '_bn' in name and '7x7' not in name:
        name = name.replace('_bn', '.bn')
    if 'conv1_7x7_s2' in name and 'bn' not in name:
        name = 'conv1_7x7_s2.' + name
    for item in modules:
        if item in name:
            name = item + '.' + name
    name = 'base_model.' + name
    return name


def convertpth2ckpt(arg):
    torch_params = torch.load(arg.model_path, map_location='cpu')
    new_params_list = []
    for name in torch_params:
        parameter = torch_params[name]
        if name == 'conv1_7x7_s2.weight':
            if arg.modality == 'Flow':
                kernel_size = parameter.size()
                new_kernel_size = kernel_size[:1] + (2 * arg.new_length,) + kernel_size[2:]
                parameter = parameter.data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_params_dict = {}
        if 'fc' in name:
            continue
        name = checkname(name)
        if parameter.numpy().dtype == 'float64':
            parameter = parameter.data.to(torch.float32)
        print(f'{name}')
        new_params_dict['name'] = name
        new_params_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(new_params_dict)
    save_checkpoint(new_params_list, 'tsn_{}.ckpt'.format(str(arg.modality).lower()))


if __name__ == '__main__':
    args = parser.parse_args()
    convertpth2ckpt(args)
    print("****" * 20)
