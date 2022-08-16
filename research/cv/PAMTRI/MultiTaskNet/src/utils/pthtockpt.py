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
"""pth --> ckpt"""
import argparse
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

def replace_self(name1, str1, str2):
    """replace"""
    return name1.replace(str1, str2)

parser = argparse.ArgumentParser(description='trans pth to ckpt')

parser.add_argument('--pth_path', type=str, default='',
                    help='pth path')
parser.add_argument('--device_target', type=str, default='cpu',
                    help='device target')

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    par_dict = torch.load(args.pth_path, map_location=args.device_target)
    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]

        name = replace_self(name, "denselayer1.", "cell_list.0.")
        name = replace_self(name, "denselayer2.", "cell_list.1.")
        name = replace_self(name, "denselayer3.", "cell_list.2.")
        name = replace_self(name, "denselayer4.", "cell_list.3.")
        name = replace_self(name, "denselayer5.", "cell_list.4.")
        name = replace_self(name, "denselayer6.", "cell_list.5.")
        name = replace_self(name, "denselayer7.", "cell_list.6.")
        name = replace_self(name, "denselayer8.", "cell_list.7.")
        name = replace_self(name, "denselayer9.", "cell_list.8.")
        name = replace_self(name, "denselayer10.", "cell_list.9.")
        name = replace_self(name, "denselayer11.", "cell_list.10.")
        name = replace_self(name, "denselayer12.", "cell_list.11.")
        name = replace_self(name, "denselayer13.", "cell_list.12.")
        name = replace_self(name, "denselayer14.", "cell_list.13.")
        name = replace_self(name, "denselayer15.", "cell_list.14.")
        name = replace_self(name, "denselayer16.", "cell_list.15.")
        name = replace_self(name, "denselayer17.", "cell_list.16.")
        name = replace_self(name, "denselayer18.", "cell_list.17.")
        name = replace_self(name, "denselayer19.", "cell_list.18.")
        name = replace_self(name, "denselayer20.", "cell_list.19.")
        name = replace_self(name, "denselayer21.", "cell_list.20.")
        name = replace_self(name, "denselayer22.", "cell_list.21.")
        name = replace_self(name, "denselayer23.", "cell_list.22.")
        name = replace_self(name, "denselayer24.", "cell_list.23.")

        name = replace_self(name, "transition1.", "transition1.features.")
        name = replace_self(name, "transition2.", "transition2.features.")
        name = replace_self(name, "transition3.", "transition3.features.")

        name = replace_self(name, "norm.1", "norm1")
        name = replace_self(name, "norm.2", "norm2")
        name = replace_self(name, "conv.1", "conv1")
        name = replace_self(name, "conv.2", "conv2")

        if name.endswith('num_batches_tracked'):
            continue
        elif (name.endswith('running_mean') or name.endswith('running_var')):
            name = replace_self(name, "running_mean", "moving_mean")
            name = replace_self(name, "running_var", "moving_variance")
        else:
            name = replace_self(name, "norm.weight", "norm.gamma")
            name = replace_self(name, "norm.bias", "norm.beta")
            name = replace_self(name, "norm0.weight", "norm0.gamma")
            name = replace_self(name, "norm0.bias", "norm0.beta")
            name = replace_self(name, "norm1.weight", "norm1.gamma")
            name = replace_self(name, "norm1.bias", "norm1.beta")
            name = replace_self(name, "norm2.weight", "norm2.gamma")
            name = replace_self(name, "norm2.bias", "norm2.beta")
            name = replace_self(name, "norm5.weight", "norm5.gamma")
            name = replace_self(name, "norm5.bias", "norm5.beta")

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'MultiTask_pretrained.ckpt')
