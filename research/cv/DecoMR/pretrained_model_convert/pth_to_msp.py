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

import os.path
import argparse

import resnet_pth
import torch.nn

import resnet_msp
import mindspore.nn


def convert_resnet(pretrained_file, result):
    resnet50_pth = resnet_pth.resnet50()
    resnet50_msp = resnet_msp.resnet50()
    if torch.cuda.is_available():
        resnet50_pth.load_state_dict(torch.load(pretrained_file), strict=False)
    else:
        resnet50_pth.load_state_dict(torch.load(pretrained_file, map_location=torch.device("cpu")), strict=False)

    p_pth_list = list()
    for p_pth in resnet50_pth.parameters():
        p_pth_list.append(p_pth.cpu().detach().numpy())

    bn_list = list()
    for m in resnet50_pth.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_list.append(m.running_mean.cpu().numpy())
            bn_list.append(m.running_var.cpu().numpy())
    p_index = 0
    bn_index = 0
    for n_msp, p_msp in resnet50_msp.parameters_and_names():
        if "moving_" not in n_msp:
            p_msp.set_data(mindspore.Tensor(p_pth_list[p_index]))
            p_index += 1
        else:
            p_msp.set_data(mindspore.Tensor(bn_list[bn_index]))
            bn_index += 1
    mindspore.save_checkpoint(resnet50_msp, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg", "resnet"], type=str)
    parser.add_argument("--pth_file", type=str, default="data/resnet50-19c8e357.pth", help="input pth file")
    parser.add_argument("--msp_file", type=str, default="data/resnet50_msp.ckpt", help="output msp file")
    args = parser.parse_args()
    if not os.path.exists(args.pth_file):
        raise FileNotFoundError(args.pth_file)
    if args.model == "resnet":
        convert_resnet(args.pth_file, args.msp_file)
    else:
        print("unknown model")
    print("success")
