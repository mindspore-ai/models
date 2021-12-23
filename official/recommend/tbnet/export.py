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
"""TB-Net evaluation."""

import os
import argparse
import numpy as np

from mindspore import context, load_checkpoint, load_param_into_net, Tensor, export

from src import tbnet, config


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Preprocess TBNet training data.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='test.csv',
        help="the csv datafile inside the dataset folder (e.g. test.csv)"
    )

    parser.add_argument(
        '--checkpoint_id',
        type=int,
        required=True,
        help="use which checkpoint(.ckpt) file to eval"
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
        choices=['Ascend'],
        help="run code on GPU"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def export_tbnet():
    """Data preprocess for inference."""
    args = get_args()

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    ckpt_path = os.path.join(home, 'checkpoints')

    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    net_config = config.TBNetConfig(config_path)

    print(f"creating TBNet from checkpoint {args.checkpoint_id} for evaluation...")
    network = tbnet.TBNet(net_config)
    param_dict = load_checkpoint(os.path.join(ckpt_path, f'tbnet_epoch{args.checkpoint_id}.ckpt'))
    load_param_into_net(network, param_dict)

    loss_net = tbnet.NetWithLossClass(network, net_config)
    train_net = tbnet.TrainStepWrap(loss_net, net_config.lr)
    train_net.set_train()
    eval_net = tbnet.PredictWithSigmoid(network)

    item = Tensor(np.ones((1,)).astype(np.int))
    rl1 = Tensor(np.ones((1, 39)).astype(np.int))
    ety = Tensor(np.ones((1, 39)).astype(np.int))
    rl2 = Tensor(np.ones((1, 39)).astype(np.int))
    his = Tensor(np.ones((1, 39)).astype(np.int))
    rate = Tensor(np.ones((1,)).astype(np.float32))
    inputs = [item, rl1, ety, rl2, his, rate]
    export(eval_net, *inputs, file_name='tbnet', file_format='MINDIR')

if __name__ == '__main__':
    export_tbnet()
