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

"""pth --> ckpt"""
import argparse
import json

from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

import torch


parser = argparse.ArgumentParser(description="trans pth to ckpt")
parser.add_argument('--pth-path', type=str, default='resnet50-19c8e357.pth', help="The path of pth file")
parser.add_argument('--ckpt-path', type=str, default='pretrained_resnet50.ckpt', help='The path to save ckpt file')
parser.add_argument('--dict-file', type=str, required=True, help='dict file')

args = parser.parse_args()

pth_dict = torch.load(args.pth_path)


with open(args.dict_file, 'r') as f:
    name_dict = json.load(f)

new_param_list = []

for pth_name, ckpt_name in name_dict.items():
    param_dict = {}
    data = pth_dict[pth_name]
    param_dict['name'] = ckpt_name
    param_dict['data'] = Tensor(data.detach().numpy())
    new_param_list.append(param_dict)


save_checkpoint(new_param_list, args.ckpt_path)
print(f'The ckpt file is saved in {args.ckpt_path}')
