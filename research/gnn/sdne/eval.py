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
"""
python eval.py
"""
import argparse

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import GraphDataset
from src import SDNE, SDNEWithLossCell, SDNELoss
from src import reconstruction_precision_k
from src import cfg

parser = argparse.ArgumentParser(description='Mindspore SDNE Training')

# Datasets
parser.add_argument('--data_url', type=str, default='', help='data path')
parser.add_argument('--dataset', type=str, default='WIKI',
                    choices=['WIKI', 'BLOGCATALOG', 'FLICKR', 'YOUTUBE', 'GRQC', 'NEWSGROUP'])

# Checkpoints
parser.add_argument('-c', '--checkpoint', required=True, type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# Device options
parser.add_argument('--device_id', type=int, default=0)

args = parser.parse_args()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)

    config = cfg[args.dataset]

    data_url = args.data_url
    if data_url == '':
        data_url = config['dataset_path']
    dataset = GraphDataset(data_url, batch=config['batch'], delimiter=config['delimiter'])
    net = SDNEWithLossCell(SDNE(dataset.get_node_size(), hidden_size=config['hidden_size']),
                           SDNELoss(alpha=config['alpha'], beta=config['beta']))

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.checkpoint)
    load_param_into_net(net, param_dict)
    index, data = dataset.get_data(config['eval']['frac'], config['eval']['use_rand'])
    idx2node_y = dataset.get_idx2node()
    idx2node_x = idx2node_y[index]
    reconstructions, vertices = net.get_reconstructions(data, idx2node_y, idx2node_x)
    reconstruction_precision_k(reconstructions, vertices, dataset.get_graph(), config['eval']['k_query'])
