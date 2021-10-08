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

import argparse
import torch

from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor, context


def pytorch2mindspore(pth_path):
    par_dict = torch.load(pth_path)
    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, 'vggm.ckpt')
    print('convert pth to ckpt finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='pth2ckpt')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--target_device', type=int, default=0)
    parser.add_argument('--pth_path', type=str, default='')
    args = parser.parse_args()
    context.set_context(device_target=args.device_target, mode=context.PYNATIVE_MODE, device_id=args.target_device)
    pytorch2mindspore(args.pth_path)
