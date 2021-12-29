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
python utils.py
"""
import numpy as np

def preprocess_nxgraph(graph):
    """
    create idx2node and node2idx

    Args:
        graph(DiGraph): the total net graph

    Returns:
        idx2node and node2idx
    """
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return np.array(idx2node, dtype=np.int32), node2idx

def read_node_label(filename, skip_head=False):
    """
    create node label

    Args:
        filename(str): the label path
        skip_head(bool): whether skip the first line

    Returns:
        node list and the corresponding label list
    """
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def reconstruction_precision_k(reconstructions, vertices, graph, k_query=None):
    """
    calculate reconstruction precision

    Args:
        reconstructions(array): net reconstruction matrix items
        vertices(array): node pairs
        graph(DiGraph): the total net graph
        k_query(list): query list

    Returns:
        Precision@K list and MAP
    """
    if k_query is None:
        k_query = [1, 2, 10, 20, 100, 200, 1000, 2000, 4000, 6000, 8000, 10000]

    def get_precisionK(max_index):
        sortedInd = np.argsort(reconstructions)[::-1]
        K = 0
        true_pred_count = 0
        prec_k_list = []
        true_size_edges = graph.number_of_edges()
        precision_list = []

        for ind in sortedInd:
            u = vertices[ind, 0]
            v = vertices[ind, 1]
            if u == v:
                continue

            K += 1
            if graph.has_edge(u, v):
                true_pred_count += 1
                if true_pred_count <= true_size_edges:
                    precision_list.append(true_pred_count / K)

            if K <= max_index:
                prec_k_list.append(true_pred_count / K)

            if true_pred_count > true_size_edges and K > max_index:
                break

        AP = np.sum(precision_list) / true_size_edges
        return prec_k_list, AP

    print('\nReconstruction Precision K ', k_query)
    precisionK_list, AP = get_precisionK(np.max(k_query))
    k_query_res = []
    for k in k_query:
        print(f"Precision@K({k})=\t{precisionK_list[k - 1]}")
        k_query_res.append(precisionK_list[k - 1])
    print('MAP : ', AP)
    return k_query_res, AP
