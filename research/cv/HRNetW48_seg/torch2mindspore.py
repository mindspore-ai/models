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
"""Transform torch model to mindspore format."""
import argparse
import torch
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint

from src.seg_hrnet import get_seg_model
from src.config import hrnetw48_config as model_config


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser(description="Transform torch model to mindspore format.")
    parser.add_argument("--pth_path", type=str, help="Torch model path.")
    parser.add_argument("--ckpt_path", type=str, help="Output checkpoint storage path.")

    return parser.parse_args()


def convert(pth_path, ckpt_path):
    """Model converter."""
    params_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    keys = list(params_dict.keys())
    print(list(params_dict.keys()))
    compare = {}
    i = 0
    while i < len(keys):
        temp = keys[i]
        if "model." in keys[i]:
            temp = temp.replace("model.", "")
            print(temp)
        if "running_mean" in temp:
            temp = temp.replace("running_mean", "moving_mean")
        if "running_var" in temp:
            temp = temp.replace("running_var", "moving_variance")
        if "weight" in temp:
            if i+2 < len(keys) and "running" in keys[i+2]:
                temp = temp.replace("weight", "gamma")
        if "bias" in temp:
            if i+2 < len(keys) and "running" in keys[i+2]:
                temp = temp.replace("bias", "beta")
        compare[keys[i]] = temp
        i += 1
    ms_params = []
    for item in compare.items():
        pair = {}
        tk = item[0]
        mk = item[1]
        if "tracked" in tk or "loss.criterion.weight" in tk:
            continue
        pair['name'] = mk
        pair['data'] = Tensor(params_dict[tk].numpy())
        ms_params.append(pair)
    save_checkpoint(ms_params, ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    convert(args.pth_path, args.ckpt_path)
    net = get_seg_model(model_config, 1000)
    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, params, strict_load=True)
    print("Transform successfully!")
