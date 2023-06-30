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
import argparse

import torch
import mindspore as ms


def pytorch2mindspore(pytorch_model_file='pretrained/mit_b0.pth',
                      mindspore_model_file="pretrained/ms_pretrained_b0.ckpt"):
    # read pth file
    par_dict = torch.load(pytorch_model_file, map_location='cpu')
    params_list = []
    for pt_name in par_dict:
        ms_param_name = get_ms_param_name_from_pt(pt_name)
        if ms_param_name is None or ms_param_name == "":
            print(f"no map found for param [{pt_name}], ignore.")
            continue
        print(f"convert pt param [{pt_name}] to ms [{ms_param_name}].")
        param_dict = {}
        parameter = par_dict[pt_name]
        param_dict['name'] = ms_param_name
        param_dict['data'] = ms.Tensor(parameter.numpy())
        params_list.append(param_dict)
    ms.save_checkpoint(params_list, mindspore_model_file)
    print(f"convert pt model to ms model success.")


def get_ms_param_name_from_pt(pt_param_name):
    ms_param_name = ''
    if pt_param_name.startswith('patch_embed') or pt_param_name.startswith('block') or pt_param_name.startswith('norm'):
        ms_param_name = pt_param_name
        if 'norm' in ms_param_name:
            ms_param_name = ms_param_name.replace('weight', 'gamma')
            ms_param_name = ms_param_name.replace('bias', 'beta')
        if 'proj' in ms_param_name:
            ms_param_name = ms_param_name.replace('proj', 'conv')
        ms_param_name = "backbone." + ms_param_name
    return ms_param_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert model param", add_help=False)
    parser.add_argument("--pt_model_path", type=str, default="pretrained/mit_b0.pth",
                        help="pytorch model path")
    parser.add_argument("--ms_model_path", type=str, default="pretrained/ms_pretrained_b0.ckpt",
                        help="mindspore model path")
    path_args, _ = parser.parse_known_args()
    pt_model_path = path_args.pt_model_path
    ms_model_path = path_args.ms_model_path
    pytorch2mindspore(pt_model_path, ms_model_path)
