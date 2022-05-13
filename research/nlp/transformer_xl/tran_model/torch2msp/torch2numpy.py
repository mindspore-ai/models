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

import argparse
import os
import pickle
import torch

# 需要1.x版本的PyTorch实现numpy的转化
parser = argparse.ArgumentParser(description='PyTorch Model Trans numpy.')
parser.add_argument('--dataset', default='enwik8',
                    help='Dataset Name.', choices=["enwik8", "text8"])
parser.add_argument('--work_dir', default="./enwik8_base.pkl", help='Directory of model param.')
args = parser.parse_args()

torch_model_path = args.work_dir
torch_param = None
if args.dataset == 'enwik8':
    torch_param = torch.load(os.path.join(torch_model_path, 'enwik8_base.pkl'))
if args.dataset == 'text8':
    torch_param = torch.load(os.path.join(torch_model_path, 'text8_base.pkl'))

if not torch_param:
    print('no torch param model.')
    exit()
torch_dict = {}
for key in torch_param.keys():
    torch_dict[key] = torch_param[key].cpu().numpy()

if args.dataset == 'enwik8':
    with open('./enwik8_base.pkl', 'wb') as f:
        pickle.dump(torch_dict, f)
if args.dataset == 'text8':
    with open('./text8_base.pkl', 'wb') as f:
        pickle.dump(torch_dict, f)
print('finish!')
