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
"""convert torch vgg16 pretrain model to mindspore VGG16 feature extractor"""

from pathlib import Path

import torch
from mindspore import Tensor
from mindspore import save_checkpoint

from model_utils.config import config


def _convert_by_k(old_k, e_num, l_num, data):
    """convert data by key"""
    new_l_num = l_num - 5 * (e_num - 1)
    new_k = old_k.replace('features', f'enc_{e_num}').replace(f'{l_num}', f'{new_l_num}')
    state_dict = {'name': new_k, 'data': Tensor(data.numpy())}
    return state_dict


def convert_ckpt(cfg):
    """convert ckpt"""
    t_state_dict = torch.load(cfg.torch_pretrained_vgg, map_location=torch.device('cpu'))

    ms_state_dict = []

    for k, v in t_state_dict.items():
        if k.startswith('classifier'):
            continue

        i = int(k.split('.')[1])
        if i < 5:
            converted_data = _convert_by_k(k, 1, i, v)
        elif 5 <= i < 10:
            converted_data = _convert_by_k(k, 2, i, v)
        elif 10 <= i < 17:
            converted_data = _convert_by_k(k, 3, i, v)
        else:
            continue
        ms_state_dict.append(converted_data)

    vgg_path = Path(cfg.torch_pretrained_vgg).resolve()
    output_path = vgg_path.parent / f'vgg16_feat_extr_ms.ckpt'
    save_checkpoint(ms_state_dict, output_path.as_posix())
    print(f'VGG16 feature extractor mindspore checkpoint saved in: {output_path}')


if __name__ == "__main__":
    convert_ckpt(config)
