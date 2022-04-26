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
"""Parse arguments"""
import argparse
import ast
from pathlib import Path
from pprint import pformat

import yaml


class Config:
    """
    Configuration namespace, convert dictionary to members.

    Args:
        cfg_dict (dict): Config parameters.
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


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser (argparse.ArgumentParser): Parent parser.
        cfg (dict): Base configuration.
        helper (dict): Helper description.
        choices (dict): Choices.
        cfg_path (str): Path to default_config.yaml.

    Returns:
        args: Parsed args from default_config.yaml.
    """
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else f"Please reference to {cfg_path}"
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path (str): Path to yaml config.

    Returns:
        cfg: Config parameters values.
        cfg_helper: Config parameters descriptions.
        cfg_choices: Config parameters choices.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs_raw = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = []
            for cf in cfgs_raw:
                cfgs.append(cf)

            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
        except ValueError("Failed to parse yaml") as err:
            raise err

    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.

    Returns:
        cfg: Merged arguments.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]

    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments.

    Returns:
        config: Parsed and merged config arguments from argparse and yaml config.
    """
    parser = argparse.ArgumentParser(description="IndexNet config.", add_help=False)
    curr_dir = Path(__file__).resolve().parent
    parser.add_argument("--config_path", type=str, default=str(curr_dir / '../../default_config.yaml'),
                        help="Path to config.")
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices)
    final_config = merge(args, default)

    return Config(final_config)


config = get_config()
