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

param = {
    'weight': 'gamma',
    'bias': 'beta',
    'running_mean': 'moving_mean',
    'running_var': 'moving_variance'
}


def pytorch2mindspore():
    """

    Returns:
        object:
    """
    par_dict = torch.load('./model-last.pth.tar', map_location='cpu')
    par_dict = par_dict['state_dict']
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        if name.endswith('num_batches_tracked'):
            continue
        name = name[7:]

        for fix in param:
            if name.endswith(fix) and (name.find('bn') != -1 or name.find('downsample.1') != -1
                                       or name.find('fc.1') != -1):
                name = name.replace(fix, param[fix])

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'reid.ckpt')

if __name__ == '__main__':
    pytorch2mindspore()
