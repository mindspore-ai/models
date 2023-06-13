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

"""export checkpoint file into mindir models"""
import argparse

import yaml
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.model import network


def get_args():
    """
    Cmd example:

    python export.py
    --config_path=./config.yaml
    """
    parser = argparse.ArgumentParser(description="Semantic human matting")
    parser.add_argument("--config_path", type=str, default="./config.yaml", help="config path")
    args = parser.parse_args()
    print(args)
    return args


def get_config_from_yaml(args):
    yaml_file = open(args.config_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()

    y = yaml.load(file_data, Loader=yaml.FullLoader)
    cfg = y["export"]
    return cfg


def export_model(cfg):
    device_id = 0
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg["device_target"])

    if cfg["device_target"] == "Ascend":
        context.set_context(device_id=device_id)

    net = network.net(stage=2)
    param_dict = load_checkpoint(cfg["ckpt_file"])
    load_param_into_net(net, param_dict)
    net.set_train(False)

    x = Tensor(np.random.uniform(-1.0, 1.0, [1, 3, 320, 320]).astype(np.float32))
    export(net, x, file_name=cfg["file_name"], file_format=cfg["file_format"])


if __name__ == "__main__":
    arguments = get_args()
    config = get_config_from_yaml(arguments)
    export_model(config)
