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

"""Parse arguments"""

import os
import ast
import argparse
import collections
from copy import deepcopy
from pprint import pformat
import yaml

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg = cfgs[0]
            elif len(cfgs) == 2:
                cfg, _ = cfgs
            elif len(cfgs) == 3:
                cfg, _, _ = cfgs
            else:
                raise ValueError("At most 3 docs (config description for help, choices) are supported in config yaml")
        except:
            raise ValueError("Failed to parse yaml")

    base = "__BASE__"
    if base in cfg:
        all_base_cfg = {}
        base_yamls = list(cfg[base])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(yaml_path), base_yaml)

            base_cfg = parse_yaml(base_yaml)
            all_base_cfg = merge_config(base_cfg, all_base_cfg)
        del cfg[base]
        return merge_config(cfg, all_base_cfg)
    return cfg


def merge_config(config, base):
    """Merge config"""
    new = deepcopy(base)
    for k, _ in config.items():
        if (k in new and isinstance(new[k], dict) and
                isinstance(config[k], collectionsAbc.Mapping)):
            new[k] = merge_config(config[k], new[k])
        else:
            if not config[k] is None and not config[k] == "":
                new[k] = config[k]
    return new


def get_train_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description="train param", add_help=False)
    parser.add_argument("--config_path", type=str, default="./config/segformer.b0.512x1024.city.yaml",
                        help="Config file path")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="run distribute")
    parser.add_argument("--run_eval", type=ast.literal_eval, default=True, help="run eval")
    parser.add_argument("--data_path", type=str, default="/cache/data/cityscapes/", help="data path")
    parser.add_argument("--pretrained_ckpt_path", type=str, default="./pretrained/ms_pretrained_b0.ckpt",
                        help="pretrained ckpt path")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument('--data_url', type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument('--train_url', type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument('--data_dir', type=str, default="/cache/data/", help="ModelArts: obs path to dataset folder")
    path_args, _ = parser.parse_known_args()
    file_cfg = parse_yaml(path_args.config_path)
    merged_config = merge_config(parser.parse_args().__dict__, file_cfg)
    return Config(merged_config)


def get_eval_config():
    parser = argparse.ArgumentParser(description="eval param", add_help=False)
    parser.add_argument("--config_path", type=str, default="./config/segformer.b0.512x1024.city.yaml",
                        help="Config file path")
    parser.add_argument("--eval_ckpt_path", type=str, default="./checkpoint/segformer_mit_b0_best.ckpt",
                        help="eval ckpt path")
    parser.add_argument("--data_path", type=str, default="/cache/data/cityscapes/", help="data path")
    path_args, _ = parser.parse_known_args()
    file_cfg = parse_yaml(path_args.config_path)
    merged_config = merge_config(parser.parse_args().__dict__, file_cfg)
    return Config(merged_config)


def get_infer_config():
    parser = argparse.ArgumentParser(description="infer param", add_help=False)
    parser.add_argument("--config_path", type=str, default="./config/segformer.b0.512x1024.city.yaml",
                        help="Config file path")
    parser.add_argument("--infer_ckpt_path", type=str, default="./checkpoint/segformer_mit_b0_best.ckpt",
                        help="infer ckpt path")
    parser.add_argument("--data_path", type=str, default="/cache/data/cityscapes/", help="data path")
    parser.add_argument("--infer_output_path", type=str, default="./infer_result/", help="infer output path")
    path_args, _ = parser.parse_known_args()
    file_cfg = parse_yaml(path_args.config_path)
    merged_config = merge_config(parser.parse_args().__dict__, file_cfg)
    return Config(merged_config)


def get_export_config():
    parser = argparse.ArgumentParser(description="export param", add_help=False)
    parser.add_argument("--config_path", type=str, default="./config/segformer.b0.512x1024.city.yaml",
                        help="Config file path")
    parser.add_argument("--export_ckpt_path", type=str, default="./checkpoint/segformer_mit_b0_best.ckpt",
                        help="export ckpt path")
    parser.add_argument("--export_format", type=str, default="MINDIR", help="export format")
    path_args, _ = parser.parse_known_args()
    file_cfg = parse_yaml(path_args.config_path)
    merged_config = merge_config(parser.parse_args().__dict__, file_cfg)
    return Config(merged_config)
