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

"""Load model"""
import os
from mindspore import load_checkpoint, load_param_into_net


def acquire_max_epoch_ckpt(cfg, stage="pre_train_t_net"):
    """
    Obtain the maximum epoch value of the .ckpt file saved in the training phase and the path of the .ckpt file
    """
    path_dir = os.path.join(cfg["saveCkpt"], stage)
    if not os.path.exists(path_dir):
        return 0, None
    list_file = os.listdir(path_dir)
    for i in range(len(list_file) - 1, -1, -1):
        if "semantic_hm_latest" not in list_file[i]:
            del list_file[i]
    list_file = sorted(list_file, key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))

    if list_file:
        cur_epoch = int(list_file[-1].split("/")[-1].split("_")[-1].split(".")[0])
        pretrain_path = os.path.join(path_dir, list_file[-1])
        return cur_epoch, pretrain_path
    return 0, None


def load_init_weight(net, init_weight_file):
    """
    Load init weight file
    """
    print("----> loading init_weight_file")
    if not os.path.exists(init_weight_file):
        print("init_weight_file [{}] is not exist.".format(init_weight_file))
    else:
        print("loading init_weight_file: {}".format(init_weight_file))
        pd = load_checkpoint(init_weight_file)
        load_param_into_net(net, pd)


def acquire_half_model(net, cfg):
    """Acquire the weight of T-Net and M-Net network"""
    dict_stage = {"pre_train_t_net": "t_net", "pre_train_m_net": "m_net"}
    if os.path.exists(cfg["init_weight"]):
        param_save = load_checkpoint(cfg["init_weight"])
    else:
        param_save = net.parameters_dict()

    cur_epoch, pretrain_path = acquire_max_epoch_ckpt(cfg, stage="pre_train_t_net")
    if cur_epoch:
        param_dict = load_checkpoint(pretrain_path)
        for key in param_dict.keys():
            if dict_stage["pre_train_t_net"] in key:
                param_save.update({key: param_dict[key]})
        print("train phase [{}] pre_model [{}] is loaded success.".format("pre_train_t_net", pretrain_path))
    else:
        print("train phase [{}] pre_model is not exist.".format("pre_train_t_net"))

    ckpt_file = os.path.join(cfg["saveCkpt"], "pre_train_m_net", "semantic_hm_best.ckpt")
    if os.path.exists(ckpt_file):
        param_dict = load_checkpoint(pretrain_path)
        for key in param_dict.keys():
            if dict_stage["pre_train_m_net"] in key:
                param_save.update({key: param_dict[key]})
        print("train phase [{}] pre_model [{}] is loaded success.".format("pre_train_m_net", pretrain_path))
    else:
        print("[{}] is not exist.".format(ckpt_file))

    return param_save


def load_pre_model(net, cfg):
    """Load pre_train model"""
    if cfg["finetuning"] is False:
        print("not load pre_model.")
        return 0

    print("----> loading pre_model")
    cur_epoch, pretrain_path = acquire_max_epoch_ckpt(cfg, stage=cfg["train_phase"])
    if cur_epoch:
        print("loading pre_model: {}".format(pretrain_path))
        param_dict = load_checkpoint(pretrain_path)
        load_param_into_net(net, param_dict)
    else:
        if cfg["train_phase"] == "end_to_end":
            param_save = acquire_half_model(net, cfg)
            load_param_into_net(net, param_save)
        else:
            load_init_weight(net, cfg["init_weight"])

    return cur_epoch
