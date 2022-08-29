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
python eval.py
"""

import argparse

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import GraphDataset
from src import SDNE, SDNEWithLossCell, SDNELoss1
from src import check_reconstruction
from src import reconstruction_precision_k
from src import cfg


parser = argparse.ArgumentParser(description='Mindspore SDNE Training')

# Datasets
parser.add_argument('--data_url', type=str, default='', help='dataset path')
parser.add_argument('--data_path', type=str, default='', help='data path')
parser.add_argument('--label_path', type=str, default='', help='label path')
parser.add_argument('--dataset', type=str, default='WIKI', choices=['WIKI', 'GRQC'])

# Checkpoints
parser.add_argument('-c', '--checkpoint', required=True, type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# Device options
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend")
parser.add_argument('--device_id', type=int, default=0)


def run_eval():
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    config = cfg[args.dataset]

    data_path = ''
    if args.data_url == '':
        data_path = args.data_path
    else:
        data_path = args.data_url + (config['data_path'] if args.data_path == '' else args.data_path)

    dataset = GraphDataset(args.dataset, data_path, batch=config['batch'], delimiter=config['delimiter'])
    net = SDNEWithLossCell(SDNE(dataset.get_node_size(), hidden_size=config['hidden_size']), SDNELoss1())

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.checkpoint)
    load_param_into_net(net, param_dict)
    _, data = dataset.get_data()
    idx2node = dataset.get_idx2node()
    embeddings = net.get_embeddings(data, config['batch'])
    if config['reconstruction']['check']:
        if args.dataset == 'WIKI':
            reconstructions, vertices = net.get_reconstructions(data, idx2node)
            reconstruction_precision_k(reconstructions, vertices, dataset.get_graph(),
                                       config['reconstruction']['k_query'])
        else:
            check_reconstruction(embeddings, dataset.get_graph(), idx2node,
                                 config['reconstruction']['k_query'])


if __name__ == "__main__":
    run_eval()
