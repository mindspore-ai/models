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
from pprint import  pformat
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


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="LEO-N5-K1_miniImageNet_config.yaml"):
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
        if item in ("dataset_name", "num_tr_examples_per_class"):
            continue
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


def get_config(get_args=False):
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description="default name", add_help=False)

    parser.add_argument("--num_tr_examples_per_class", type=int,
                        default=5,
                        help="num_tr_examples_per_class")
    parser.add_argument("--dataset_name", type=str,
                        default="miniImageNet",
                        help="dataset_name")
    path_args, _ = parser.parse_known_args()
    config_name = "LEO-N5-K" + str(path_args.num_tr_examples_per_class) \
                  + "_" + path_args.dataset_name + "_config.yaml"
    config_path = os.path.join(os.path.abspath(os.path.join(__file__, "../..")), "config", config_name)


    default, helper, choices = parse_yaml(config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=config_path)
    if get_args:
        return args
    final_config = merge(args, default)
    return Config(final_config)


def get_init_config():
    config = {}
    config["device_target"] = get_config().device_target
    config["device_num"] = get_config().device_num
    config["data_path"] = get_config().data_path
    config["save_path"] = get_config().save_path
    config["ckpt_file"] = get_config().ckpt_file
    config["dataset_name"] = get_config().dataset_name
    config["embedding_crop"] = get_config().embedding_crop
    config["train_on_val"] = get_config().train_on_val
    config["total_examples_per_class"] = 600

    return config


def get_inner_model_config():
    """Returns the config used to initialize LEO model."""
    config = {}
    config["inner_unroll_length"] = get_config().inner_unroll_length
    config["finetuning_unroll_length"] = get_config().finetuning_unroll_length
    config["num_latents"] = get_config().num_latents
    config["inner_lr_init"] = get_config().inner_lr_init
    config["finetuning_lr_init"] = get_config().finetuning_lr_init
    config["dropout_rate"] = get_config().dropout_rate
    config["kl_weight"] = get_config().kl_weight
    config["encoder_penalty_weight"] = get_config().encoder_penalty_weight
    config["l2_penalty_weight"] = get_config().l2_penalty_weight
    config["orthogonality_penalty_weight"] = get_config().orthogonality_penalty_weight

    return config


def get_outer_model_config():
    """Returns the outer config file for N-way K-shot classification tasks."""
    config = {}
    config["num_classes"] = get_config().num_classes
    config["num_tr_examples_per_class"] = get_config().num_tr_examples_per_class
    config["num_val_examples_per_class"] = get_config().num_val_examples_per_class
    config["metatrain_batch_size"] = get_config().metatrain_batch_size
    config["metavalid_batch_size"] = get_config().metavalid_batch_size
    config["metatest_batch_size"] = get_config().metatest_batch_size
    config["num_steps_limit"] = get_config().num_steps_limit
    config["outer_lr"] = get_config().outer_lr
    config["gradient_threshold"] = get_config().gradient_threshold
    config["gradient_norm_threshold"] = get_config().gradient_norm_threshold
    config["total_steps"] = get_config().total_steps

    return config
