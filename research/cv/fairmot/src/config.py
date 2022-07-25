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
"""
config
"""
import os
import argparse
import ast
from pprint import pformat
import yaml


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


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
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
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
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
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
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
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
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
        cfg[item] = args_var[item]
    return cfg


class Opts:
    """
    parameter configuration
    """
    def __init__(self, args=''):
        self.config = self.init(args)

    def parse(self, args=''):
        """parameter parse"""
        parser = argparse.ArgumentParser(description="default name", add_help=False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, "../default_config.yaml"),
                            help="Config file path")
        path_args, _ = parser.parse_known_args()
        default, helper, choices = parse_yaml(path_args.config_path)
        args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices,
                                 cfg_path=path_args.config_path)
        default = Config(merge(args, default))

        default.fix_res = not default.keep_res
        print('Fix size testing.' if default.fix_res else 'Keep resolution testing.')
        default.reg_offset = not default.not_reg_offset

        if default.head_conv == -1:  # init default head_conv
            default.head_conv = 256 if 'dla' in default.arch else 256
        default.pad = 31
        default.num_stacks = 1

        return default

    def update_dataset_info_and_set_heads(self, opt, dataset):
        """update dataset info and set heads"""
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        opt.heads = {'hm': opt.num_classes,
                     'wh': 2 if not opt.ltrb else 4,
                     'id': opt.reid_dim}
        if opt.reg_offset:
            opt.heads.update({'reg': 2})
        opt.nID = dataset.nID
        opt.img_size = (1088, 608)
        print('heads', opt.heads)
        return opt

    def init(self, args=''):
        """opt init"""
        default_dataset_info = {
            'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde', 'nID': 14455},
        }

        class Struct:
            """opt struct"""
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info['mot'])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt

    def get_config(self):
        return self.config
