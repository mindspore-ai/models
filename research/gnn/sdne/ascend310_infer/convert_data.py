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
python convert_data.py
"""
import argparse
import networkx as nx

parser = argparse.ArgumentParser(description='Mindspore SDNE Training')
parser.add_argument('--data_url', type=str, default='', help='dataset path')
parser.add_argument('--output_file', type=str, default='output', help='dataset path')
args = parser.parse_args()

if __name__ == "__main__":
    graph = nx.read_edgelist(args.data_url, create_using=nx.DiGraph(),
                             delimiter=' ', data=[('weight', float)], nodetype=int)
    node2idx = {}
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        node_size += 1

    with open(args.output_file, 'w') as output:
        with open(args.data_url, 'r') as source:
            line = source.readline()
            while line:
                nodes = line.strip().split()
                output.write(str(node2idx[int(nodes[0])]) + ' ' + str(node2idx[int(nodes[1])]) + '\n')
                line = source.readline()
