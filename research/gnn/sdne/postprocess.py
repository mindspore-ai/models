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
python postprocess.py
"""
import argparse
import numpy as np

from src import GraphDataset
from src import check_reconstruction
from src import reconstruction_precision_k
from src import cfg

parser = argparse.ArgumentParser(description='Mindspore SDNE 310 POSTPROCESS')
parser.add_argument('--data_path', type=str, default='', help='data path')
parser.add_argument('--emb_file', type=str, default='embeddings.txt', help='embeddings file')
parser.add_argument('--rec_file', type=str, default='reconstructions.txt', help='reconstructions file')
parser.add_argument('--dataset', type=str, default='WIKI', help='dataset')
args = parser.parse_args()

if __name__ == "__main__":
    config = cfg[args.dataset]
    dataset = GraphDataset(args.dataset, args.data_path, batch=config['batch'], delimiter=config['delimiter'])
    node_size = dataset.get_node_size()
    idx2node = dataset.get_idx2node()
    if args.dataset == 'WIKI':
        reconstructions = np.zeros(node_size * node_size, dtype=float)
        vertices = np.zeros((node_size * node_size, 2), dtype=int)
        cnt = 0
        with open(args.rec_file, 'r') as rec:
            line = rec.readline()
            while line:
                items = line.split()
                vertices[cnt][0] = idx2node[int(items[0])]
                vertices[cnt][1] = idx2node[int(items[1])]
                reconstructions[cnt] = float(items[2])
                line = rec.readline()
                cnt += 1
        reconstruction_precision_k(reconstructions, vertices, dataset.get_graph(),
                                   config['reconstruction']['k_query'])
    else:
        embeddings = []
        with open(args.emb_file, 'r') as emb:
            line = emb.readline()
            while line:
                items = line.split()
                embeddings.append(list(map(float, items)))
                line = emb.readline()
        embeddings = np.array(embeddings, dtype=float)
        check_reconstruction(embeddings, dataset.get_graph(), idx2node,
                             config['reconstruction']['k_query'])
