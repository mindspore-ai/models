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

import torch

from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

from src.c3d_model import C3D


def torch_to_mindspore(torch_file_path, mindspore_file_path):
    ckpt = torch.load(torch_file_path, map_location=torch.device('cpu'))

    new_params_list = []
    for _, v in ckpt.items():
        new_params_list.append(v.numpy())

    mindspore_params_list = []
    network = C3D(num_classes=487)
    for v, k in zip(new_params_list, network.parameters_dict().keys()):
        if 'fc8' in k:
            continue
        mindspore_params_list.append({'name': k, 'data': Tensor.from_numpy(v)})

    save_checkpoint(mindspore_params_list, mindspore_file_path)
    print('convert pytorch ckpt file to mindspore ckpt file ok !')


if __name__ == '__main__':
    torch_ckpt_file_path = sys.argv[1]
    mindspore_ckpt_file_path = sys.argv[2]
    torch_to_mindspore(torch_ckpt_file_path, mindspore_ckpt_file_path)
