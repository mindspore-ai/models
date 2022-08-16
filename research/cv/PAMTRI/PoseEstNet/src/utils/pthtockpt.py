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

        name = replace_self(name, "downsample", "downSample")
        name = replace_self(name, "transition2.2", "transition2.0")
        name = replace_self(name, "transition3.3", "transition3.0")

        if name.endswith('num_batches_tracked') or name.startswith('head') or name.startswith('fc'):
            continue
        elif (name.endswith('running_mean') or name.endswith('running_var')):
            name = replace_self(name, "running_mean", "moving_mean")
            name = replace_self(name, "running_var", "moving_variance")
        else:
            name = replace_self(name, "bn1.weight", "bn1.gamma")
            name = replace_self(name, "bn1.bias", "bn1.beta")
            name = replace_self(name, "bn2.weight", "bn2.gamma")
            name = replace_self(name, "bn2.bias", "bn2.beta")
            name = replace_self(name, "bn3.weight", "bn3.gamma")
            name = replace_self(name, "bn3.bias", "bn3.beta")

            name = replace_self(name, "downSample.1.weight", "downSample.1.gamma")
            name = replace_self(name, "downSample.1.bias", "downSample.1.beta")

            name = replace_self(name, "fuse_layers.0.1.1.weight", "fuse_layers.0.1.1.gamma")
            name = replace_self(name, "fuse_layers.0.1.1.bias", "fuse_layers.0.1.1.beta")
            name = replace_self(name, "fuse_layers.0.2.1.weight", "fuse_layers.0.2.1.gamma")
            name = replace_self(name, "fuse_layers.0.2.1.bias", "fuse_layers.0.2.1.beta")
            name = replace_self(name, "fuse_layers.0.3.1.weight", "fuse_layers.0.3.1.gamma")
            name = replace_self(name, "fuse_layers.0.3.1.bias", "fuse_layers.0.3.1.beta")
            name = replace_self(name, "fuse_layers.1.2.1.weight", "fuse_layers.1.2.1.gamma")
            name = replace_self(name, "fuse_layers.1.2.1.bias", "fuse_layers.1.2.1.beta")
            name = replace_self(name, "fuse_layers.1.3.1.weight", "fuse_layers.1.3.1.gamma")
            name = replace_self(name, "fuse_layers.1.3.1.bias", "fuse_layers.1.3.1.beta")
            name = replace_self(name, "fuse_layers.2.3.1.weight", "fuse_layers.2.3.1.gamma")
            name = replace_self(name, "fuse_layers.2.3.1.bias", "fuse_layers.2.3.1.beta")
            name = replace_self(name, "fuse_layers.1.0.1.1.weight", "fuse_layers.1.0.1.1.gamma")
            name = replace_self(name, "fuse_layers.1.0.1.1.bias", "fuse_layers.1.0.1.1.beta")
            name = replace_self(name, "fuse_layers.1.0.0.1.weight", "fuse_layers.1.0.0.1.gamma")
            name = replace_self(name, "fuse_layers.1.0.0.1.bias", "fuse_layers.1.0.0.1.beta")
            name = replace_self(name, "fuse_layers.2.0.0.1.weight", "fuse_layers.2.0.0.1.gamma")
            name = replace_self(name, "fuse_layers.2.0.0.1.bias", "fuse_layers.2.0.0.1.beta")
            name = replace_self(name, "fuse_layers.2.0.1.1.weight", "fuse_layers.2.0.1.1.gamma")
            name = replace_self(name, "fuse_layers.2.0.1.1.bias", "fuse_layers.2.0.1.1.beta")
            name = replace_self(name, "fuse_layers.2.1.0.1.weight", "fuse_layers.2.1.0.1.gamma")
            name = replace_self(name, "fuse_layers.2.1.0.1.bias", "fuse_layers.2.1.0.1.beta")
            name = replace_self(name, "fuse_layers.3.0.0.1.weight", "fuse_layers.3.0.0.1.gamma")
            name = replace_self(name, "fuse_layers.3.0.0.1.bias", "fuse_layers.3.0.0.1.beta")
            name = replace_self(name, "fuse_layers.3.0.1.1.weight", "fuse_layers.3.0.1.1.gamma")
            name = replace_self(name, "fuse_layers.3.0.1.1.bias", "fuse_layers.3.0.1.1.beta")
            name = replace_self(name, "fuse_layers.3.0.2.1.weight", "fuse_layers.3.0.2.1.gamma")
            name = replace_self(name, "fuse_layers.3.0.2.1.bias", "fuse_layers.3.0.2.1.beta")
            name = replace_self(name, "fuse_layers.3.1.0.1.weight", "fuse_layers.3.1.0.1.gamma")
            name = replace_self(name, "fuse_layers.3.1.0.1.bias", "fuse_layers.3.1.0.1.beta")
            name = replace_self(name, "fuse_layers.3.1.1.1.weight", "fuse_layers.3.1.1.1.gamma")
            name = replace_self(name, "fuse_layers.3.1.1.1.bias", "fuse_layers.3.1.1.1.beta")
            name = replace_self(name, "fuse_layers.3.2.0.1.weight", "fuse_layers.3.2.0.1.gamma")
            name = replace_self(name, "fuse_layers.3.2.0.1.bias", "fuse_layers.3.2.0.1.beta")

            name = replace_self(name, "transition1.0.1.weight", "transition1.0.1.gamma")
            name = replace_self(name, "transition1.0.1.bias", "transition1.0.1.beta")
            name = replace_self(name, "transition1.1.0.1.weight", "transition1.1.0.1.gamma")
            name = replace_self(name, "transition1.1.0.1.bias", "transition1.1.0.1.beta")

            name = replace_self(name, "transition2.0.0.1.weight", "transition2.0.0.1.gamma")
            name = replace_self(name, "transition2.0.0.1.bias", "transition2.0.0.1.beta")

            name = replace_self(name, "transition3.0.0.1.weight", "transition3.0.0.1.gamma")
            name = replace_self(name, "transition3.0.0.1.bias", "transition3.0.0.1.beta")

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'PoseEstNet_pretrained.ckpt')
