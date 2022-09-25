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
import yaml
from addict import Dict


def load_config(file_path):
    """
    Load config file.
    Args:
        file_path (str): Path of the config file.
    Returns: configs
    """
    configs = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return configs

def merge(args, config):
    """
    Merge command arguments and base configs from yaml file.

    Args:
        args: Command arguments.
        config: configuration in config file.
    """
    args_var = vars(args)
    for item in args_var:
        config[item] = args_var[item]

    return Dict(config)
