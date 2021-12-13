# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""convert pre-trained .pth file to .ckpt file"""

import torch
from mindspore import Tensor, save_checkpoint


cached_file = 'osnet_x1_0_imagenet.pth'
state_dict = torch.load(cached_file)
param_list = []
bn_param_names = []
bn_param_values = []
bn_count = 0

for k, v in state_dict.items():
    strlist = k.split('.')
    level = []
    for value in strlist:
        level.append(value)
    if level[-2].startswith('conv') or level[-2].startswith('fc'):
        param_dict = {}
        parameter = v
        param_dict['name'] = k
        param_dict['data'] = Tensor(parameter.numpy())
        param_list.append(param_dict)
    elif level[-2] == 'bn' or k.startswith('fc.1'):
        if level[-1] == 'num_batches_tracked':
            continue
        else:
            if bn_count == 3:
                bn_param_names.append(k)
                bn_param_values.append(v)
                name_levels = bn_param_names[0].rsplit('.', 1)
                prefix = name_levels[0]
                param_dict = {}
                # moving_mean
                parameter = bn_param_values[2]
                param_dict['name'] = prefix+'.moving_mean'
                param_dict['data'] = Tensor(parameter.numpy())
                param_list.append(param_dict)
                # moving_variance
                param_dict = {}
                parameter = bn_param_values[3]
                param_dict['name'] = prefix + '.moving_variance'
                param_dict['data'] = Tensor(parameter.numpy())
                param_list.append(param_dict)
                # gamma
                param_dict = {}
                parameter = bn_param_values[0]
                param_dict['name'] = prefix + '.gamma'
                param_dict['data'] = Tensor(parameter.numpy())
                param_list.append(param_dict)
                # beta
                param_dict = {}
                parameter = bn_param_values[1]
                param_dict['name'] = prefix + '.beta'
                param_dict['data'] = Tensor(parameter.numpy())
                param_list.append(param_dict)

                bn_param_names = []
                bn_param_values = []
                bn_count = 0

            else:
                bn_param_names.append(k)
                bn_param_values.append(v)
                bn_count += 1

    elif level[0] == 'fc' and level[1] == '0':
        param_dict = {}
        parameter = v
        param_dict['name'] = k
        param_dict['data'] = Tensor(parameter.numpy())
        param_list.append(param_dict)

    elif level[0] == 'classifier':
        #continue;
        param_dict = {}
        parameter = v
        param_dict['name'] = k
        param_dict['data'] = Tensor(parameter.numpy())
        param_list.append(param_dict)

save_checkpoint(param_list, 'init_osnet.ckpt')
