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

params = {
    'features.features.0.weight': 'featureExtract.0.weight',
    'features.features.0.bias': 'featureExtract.0.bias',
    'features.features.1.weight': 'featureExtract.1.gamma',
    'features.features.1.bias': 'featureExtract.1.beta',
    'features.features.1.running_mean': 'featureExtract.1.moving_mean',
    'features.features.1.running_var': 'featureExtract.1.moving_variance',
    'features.features.4.weight': 'featureExtract.4.weight',
    'features.features.4.bias': 'featureExtract.4,bias',
    'features.features.5.weight': 'featureExtract.5.gamma',
    'features.features.5.bias': 'featureExtract.5.beta',
    'features.features.5.running_mean': 'featureExtract.5.moving_mean',
    'features.features.5.running_var': 'featureExtract.5.moving_variance',
    'features.features.8.weight': 'featureExtract.8.weight',
    'features.features.8.bias': 'featureExtract.8.bias',
    'features.features.9.weight': 'featureExtract.9.gamma',
    'features.features.9.bias': 'featureExtract.9.beta',
    'features.features.9.running_mean': 'featureExtract.9.moving_mean',
    'features.features.9.running_var': 'featureExtract.9.moving_variance',
    'features.features.11.weight': 'featureExtract.11.weight',
    'features.features.11.bias': 'featureExtract.11.bias',
    'features.features.12.weight': 'featureExtract.12.gamma',
    'features.features.12.bias': 'featureExtract.12.beta',
    'features.features.12.running_mean': 'featureExtract.12.moving_mean',
    'features.features.12.running_var': 'featureExtract.12.moving_variance',
    'features.features.14.weight': 'featureExtract.14.weight',
    'features.features.14.bias': 'featureExtract.14.bias',
    'features.features.15.weight': 'featureExtract.15.gamma',
    'features.features.15.bias': 'featureExtract.15.beta',
    'features.features.15.running_mean': 'featureExtract.15.moving_mean',
    'features.features.15.running_var': 'featureExtract.15.moving_variance',
}


def convertpthtockpt(arg):
    torch_param_dict = torch.load(arg.model_path, map_location='cpu')
    new_param_list = []
    for name in torch_param_dict:
        ms_param_dict = {}
        torch_param_data = torch_param_dict[name]
        ms_param_dict['name'] = params[name]
        print(f'{name}----->{params[name]}')
        ms_param_dict['data'] = Tensor(torch_param_data.numpy())
        new_param_list.append(ms_param_dict)
    save_checkpoint(new_param_list, 'ms_alexnet.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mindspore SiameseRPN model convert')
    parser.add_argument('--model_path', default=False, type=str, help='torch model path')
    args = parser.parse_args()
    convertpthtockpt(args)
