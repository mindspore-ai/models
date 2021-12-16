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
"""
generate ckpt from pth.
"""
import argparse
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--pth_path', type=str,
                    help='pth path, eg:./data/example.pth')
parser.add_argument('--ckpt_path', type=str,
                    help='ckpt path, eg:./data/pretrain.ckpt')
args_opt = parser.parse_args()


if __name__ == '__main__':
    # pth2ckpt(args_opt.pth_path, args_opt.ckpt_path)
    params_dict1 = torch.load(args_opt.pth_path, map_location='cpu')
    state_dict1 = params_dict1['state_dict']
    new_param_list = []
    for name in state_dict1:
        param_dict = {}
        parameter = state_dict1[name]
        if name.endswith('bn1.bias'):
            name = name[:name.rfind('bn1.bias')]
            name = name + 'bn1.bn2d.beta'
        elif name.endswith('bn1.weight'):
            name = name[:name.rfind('bn1.weight')]
            name = name + 'bn1.bn2d.gamma'
        elif name.endswith('bn2.weight'):
            name = name[:name.rfind('bn2.weight')]
            name = name + 'bn2.bn2d.gamma'
        elif name.endswith('bn2.bias'):
            name = name[:name.rfind('bn2.bias')]
            name = name + 'bn2.bn2d.beta'
        elif name.endswith('bn3.bias'):
            name = name[:name.rfind('bn3.bias')]
            name = name + 'bn3.bn2d.beta'
        elif name.endswith('bn3.weight'):
            name = name[:name.rfind('bn3.weight')]
            name = name + 'bn3.bn2d.gamma'
        elif name.endswith('bn1.running_mean'):
            name = name[:name.rfind('bn1.running_mean')]
            name = name + 'bn1.bn2d.moving_mean'
        elif name.endswith('bn1.running_var'):
            name = name[:name.rfind('bn1.running_var')]
            name = name + 'bn1.bn2d.moving_variance'
        elif name.endswith('bn2.running_mean'):
            name = name[:name.rfind('bn2.running_mean')]
            name = name + 'bn2.bn2d.moving_mean'
        elif name.endswith('bn2.running_var'):
            name = name[:name.rfind('bn2.running_var')]
            name = name + 'bn2.bn2d.moving_variance'
        elif name.endswith('bn3.running_mean'):
            name = name[:name.rfind('bn3.running_mean')]
            name = name + 'bn3.bn2d.moving_mean'
        elif name.endswith('bn3.running_var'):
            name = name[:name.rfind('bn3.running_var')]
            name = name + 'bn3.bn2d.moving_variance'
        elif name.endswith('downsample.1.running_mean'):
            name = name[:name.rfind('downsample.1.running_mean')]
            name = name + 'downsample.1.bn2d.moving_mean'
        elif name.endswith('downsample.1.running_var'):
            name = name[:name.rfind('downsample.1.running_var')]
            name = name + 'downsample.1.bn2d.moving_variance'
        elif name.endswith('downsample.0.weight'):
            name = name[:name.rfind('downsample.0.weight')]
            name = name + 'downsample.0.weight'
        elif name.endswith('downsample.1.bias'):
            name = name[:name.rfind('downsample.1.bias')]
            name = name + 'downsample.1.bn2d.beta'
        elif name.endswith('downsample.1.weight'):
            name = name[:name.rfind('downsample.1.weight')]
            name = name + 'downsample.1.bn2d.gamma'
        elif name.endswith('num_batches_tracked'):
            continue
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_param_list.append(param_dict)
    save_checkpoint(new_param_list, args_opt.ckpt_path)
