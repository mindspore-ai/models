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

import os
import ast
import argparse
from pprint import pprint, pformat
import yaml

_config_path = "../config/default_gpu.yaml"

def path_constructor(loader: yaml.FullLoader, node) -> str: # : yaml.nodes.SequenceNode
    """Construct a path."""
    if isinstance(node, yaml.nodes.SequenceNode):
        return os.path.join(*loader.construct_sequence(node))
    return loader.construct_scalar(node)

def get_custom_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.FullLoader
    loader.add_constructor("!Path", path_constructor)
    return loader

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

    def __getitem__(self, key):
        return getattr(self, key)


def parse_cli_subdict(parser, cfg, prefix, helper=None, choices=None, cfg_path=_config_path):

    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if isinstance(cfg[item], dict):
            parser = parse_cli_subdict(parser, cfg[item], ".".join((prefix, str(item))), helper, choices, cfg_path)
        elif isinstance(cfg[item], list):
            pass
        else:
            _item = prefix + '.' + item
            help_description = helper[_item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[_item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + _item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + _item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)

    return parser

def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path=_config_path):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if isinstance(cfg[item], list):
            pass
        elif isinstance(cfg[item], dict):
            parser = parse_cli_subdict(parser, cfg[item], str(item), helper, choices, cfg_path)
        else:
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
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
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=get_custom_loader())
            cfgs = [x for x in cfgs]
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
                raise ValueError("At most 3 docs (config description for help, choices) are supported in config yaml")
            print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        parts = item.split('.')
        if len(parts) == 1:
            cfg[item] = args_var[item]
        elif len(parts) == 2:
            cfg[parts[0]][parts[1]] = args_var[item]
        elif len(parts) == 3:
            cfg[parts[0]][parts[1]][parts[2]] = args_var[item]
        elif len(parts) == 4:
            cfg[parts[0]][parts[1]][parts[2]][parts[3]] = args_var[item]
        else:
            raise ValueError(f"Too many nesting levels: {len(parts)}. Maximum available: {4}")

    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, \
        "../../defalut_config.yaml"), help="Config file path")
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    pprint(default)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    return Config(final_config)

config = get_config()
