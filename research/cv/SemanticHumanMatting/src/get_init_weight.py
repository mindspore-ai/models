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

"""Get initial weight from torch"""
import os
import random
import numpy as np
import torch
import mindspore
from model.network import net


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pytorch2mindspore(ckpt_path="./init_weight.pth", save_ckpt="./init_weight.ckpt"):
    """Convert weight into mindspore from torch"""
    par_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))

    nums = 0
    new_params_list = []
    dict_bn = {
        "0": ["weight", "gamma"],
        "1": ["bias", "beta"],
        "2": ["running_mean", "moving_mean"],
        "3": ["running_var", "moving_variance"],
    }
    list_dict = list(par_dict)
    n = len(list_dict)
    for idx, name in enumerate(list_dict):
        parameter = par_dict[name]
        if nums:
            nums -= 1
            continue
        print("========================py_name", name)
        # nn.BatchNorm2d
        # weight——gamma, bias——beta, running_mean——moving_mean, running_var——moving_variance
        if (
            idx + 4 < n
            and name.endswith(".weight")
            and list_dict[idx + 1].endswith(".bias")
            and list_dict[idx + 2].endswith(".running_mean")
            and list_dict[idx + 3].endswith(".running_var")
            and list_dict[idx + 4].endswith(".num_batches_tracked")
        ):
            nums = 3
            for i in range(4):
                name = list_dict[idx + i]
                parameter = par_dict[name]
                name = name[: name.rfind(dict_bn[str(i)][0])]
                name = name + dict_bn[str(i)][1]
                new_params_list.append({"name": name, "data": mindspore.Tensor(parameter.numpy())})
        else:
            new_params_list.append({"name": name, "data": mindspore.Tensor(parameter.numpy())})

    mindspore.save_checkpoint(new_params_list, save_ckpt)


if __name__ == "__main__":
    set_random_seed(9527)
    file_path = os.path.dirname(os.path.abspath(__file__))

    # acquire torch network initialize weight
    net_work = net()
    torch.save(net_work.state_dict(), "./init_weight.pth")

    # convert weight into mindspore from torch
    ckpt_file = os.path.join(file_path, "init_weight.pth")
    save_file = os.path.join(file_path, "init_weight.ckpt")
    pytorch2mindspore(ckpt_path=ckpt_file, save_ckpt=save_file)
