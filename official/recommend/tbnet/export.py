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
"""export."""

import os
import argparse
import math
import numpy as np

from mindspore import context, load_checkpoint, load_param_into_net, Tensor, export

from src import tbnet, config


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Export.')

    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        default='',
        help="json file for dataset"
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help="use which checkpoint(.ckpt) file to export"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--device_target',
        type=str,
        required=False,
        default='Ascend',
        choices=['Ascend', 'GPU'],
        help="run code on platform"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    parser.add_argument(
        '--file_name',
        type=str,
        default='tbnet',
        help="model name."
    )

    parser.add_argument(
        '--file_format',
        type=str,
        default='MINDIR',
        choices=['MINDIR', 'AIR'],
        help="model format."
    )
    return parser.parse_args()


def export_tbnet():
    """Data preprocess for inference."""
    args = get_args()

    config_path = args.config_path
    ckpt_path = args.checkpoint_path
    if not os.path.exists(config_path):
        raise ValueError("please check the config path.")
    if not os.path.exists(ckpt_path):
        raise ValueError("please check the checkpoint path.")

    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    net_config = config.TBNetConfig(config_path)
    if args.device_target == 'Ascend':
        net_config.per_item_paths = math.ceil(net_config.per_item_paths / 16) * 16
        net_config.embedding_dim = math.ceil(net_config.embedding_dim / 16) * 16
    network = tbnet.TBNet(net_config)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    eval_net = tbnet.PredictWithSigmoid(network)

    item = Tensor(np.ones((1,)).astype(np.int))
    rl1 = Tensor(np.ones((1, net_config.per_item_paths)).astype(np.int))
    ety = Tensor(np.ones((1, net_config.per_item_paths)).astype(np.int))
    rl2 = Tensor(np.ones((1, net_config.per_item_paths)).astype(np.int))
    his = Tensor(np.ones((1, net_config.per_item_paths)).astype(np.int))
    rate = Tensor(np.ones((1,)).astype(np.float32))
    inputs = [item, rl1, ety, rl2, his, rate]
    export(eval_net, *inputs, file_name=args.file_name, file_format=args.file_format)


if __name__ == '__main__':
    export_tbnet()
