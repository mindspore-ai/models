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
""" pth2ckpt. """

import sys
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint


def load_inception_ckpt():
    par_dict = torch.load(sys.argv[1])
    new_params_list = []
    i = 1
    for name in par_dict:
        if name.find('num_batches_tracked') != -1:
            continue
        print(i, name)
        i += 1
        param_dict = {}
        parameter = par_dict[name]
        name = name.replace('_3x3.', '.', 1)
        name = name.replace('running_mean', 'moving_mean', 1)
        name = name.replace('running_var', 'moving_variance', 1)
        name = name.replace('bn.bias', 'bn.beta', 1)
        name = name.replace('bn.weight', 'bn.gamma', 1)
        name = name.replace('_1x1.', '.', 1)
        name = name.replace('branch1x1', 'branch0', 1)
        name = name.replace('branch5x5_1', 'branch1.0', 1)
        name = name.replace('branch5x5_2', 'branch1.1', 1)
        name = name.replace('branch_pool', 'branch_pool.1', 1)
        name = name.replace('branch3x3.', 'branch0.', 1)
        name = name.replace('branch7x7_1', 'branch1.0', 1)
        name = name.replace('branch7x7_2', 'branch1.1', 1)
        name = name.replace('branch7x7_3', 'branch1.2', 1)
        name = name.replace('branch7x7dbl_1', 'branch2.0', 1)
        name = name.replace('branch7x7dbl_2', 'branch2.1', 1)
        name = name.replace('branch7x7dbl_3', 'branch2.2', 1)
        name = name.replace('branch7x7dbl_4', 'branch2.3', 1)
        name = name.replace('branch7x7dbl_5', 'branch2.4', 1)
        if name.find('Mixed_7b') != -1 or name.find('Mixed_7c') != -1:
            name = name.replace('branch3x3_1', 'branch1', 1)
        else:
            name = name.replace('branch3x3_1', 'branch0.0', 1)
        if name.find('Mixed_6a') != -1:
            name = name.replace('branch3x3dbl_1', 'branch1.0', 1)
            name = name.replace('branch3x3dbl_2', 'branch1.1', 1)
            name = name.replace('branch3x3dbl_3.', 'branch1.2.', 1)
        else:
            name = name.replace('branch3x3dbl_1', 'branch2.0', 1)
            name = name.replace('branch3x3dbl_2', 'branch2.1', 1)
            name = name.replace('branch3x3dbl_3.', 'branch2.2.', 1)
        name = name.replace('branch3x3_2.', 'branch0.1.', 1)
        name = name.replace('branch3x3_2a', 'branch1_a', 1)
        name = name.replace('branch3x3_2b', 'branch1_b', 1)
        name = name.replace('branch3x3dbl_3a', 'branch2_a', 1)
        name = name.replace('branch3x3dbl_3b', 'branch2_b', 1)

        name = name.replace('branch7x7x3_1', 'branch1.0', 1)
        name = name.replace('branch7x7x3_2', 'branch1.1', 1)
        name = name.replace('branch7x7x3_3', 'branch1.2', 1)
        name = name.replace('branch7x7x3_4', 'branch1.3', 1)
        print("----------------", name)
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, './inception/inception_pid.ckpt')

if __name__ == '__main__':
    load_inception_ckpt()
