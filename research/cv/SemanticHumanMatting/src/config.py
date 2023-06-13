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

"""Parse config file and acquire update config option"""
import os
import argparse
import yaml


def get_device_num():
    device_num = os.getenv("RANK_SIZE", "1")
    return int(device_num)


def get_args():
    """
    Cmd example:

    python3 train.py
    --yaml_path=../shm_mindspore/config.yaml
    --data_url=../semantic_hm_mindspore/data/datasets_debug
    --train_url=../output
    --init_weight=./pre_train_model/init_weight.ckpt
    """
    # add argument
    parser = argparse.ArgumentParser(description="semantic human matting !")
    parser.add_argument("--yaml_path", type=str, default=None, help="config path")
    parser.add_argument("--data_url", type=str, default=None, help="dataset path")
    parser.add_argument("--train_url", type=str, default=None, help="train output path")
    parser.add_argument("--init_weight", type=str, default="./init_weight.ckpt", help="init weight path, optional")
    args = parser.parse_args()
    print(args)
    return args


def print_yaml(args):
    """Print yaml file"""
    try:
        fd = open(args.yaml_path, "r", encoding="utf-8")
        content = fd.read()
        print(content)
        fd.close()
        return content
    except OSError:
        raise ValueError("open config.yaml fail")


def get_config_from_yaml(args):
    """Acquire the content of the yaml file"""
    print("------------------------------config.yaml------------------------------")
    content = print_yaml(args)
    print("-----------------------------------------------------------------------")
    device_num = get_device_num()

    if args.train_url is None:
        args.train_url = "./"

    y = yaml.load(content, Loader=yaml.FullLoader)
    for key in y.keys():
        if key in ["pre_train_t_net", "pre_train_m_net", "end_to_end"]:
            cfg = y[key]
            cfg["dataDir"] = args.data_url
            cfg["init_weight"] = args.init_weight

            if device_num != 1:
                cfg["saveCkpt"] = os.path.join(args.train_url, "distribute", y["ckpt_version"])
            else:
                cfg["saveCkpt"] = os.path.join(args.train_url, "single", y["ckpt_version"])

            y[key] = cfg

    if device_num != 1:
        y["saveIRGraph"] = os.path.join(args.train_url, "distribute", "ir_graph")
    else:
        y["saveIRGraph"] = os.path.join(args.train_url, "single", "ir_graph")

    return y


def update_config(cfg):
    cfg["pre_train_t_net"]["rank"] = cfg["rank"]
    cfg["pre_train_t_net"]["group_size"] = cfg["group_size"]
    cfg["pre_train_m_net"]["rank"] = cfg["rank"]
    cfg["pre_train_m_net"]["group_size"] = cfg["group_size"]
    cfg["end_to_end"]["rank"] = cfg["rank"]
    cfg["end_to_end"]["group_size"] = cfg["group_size"]
