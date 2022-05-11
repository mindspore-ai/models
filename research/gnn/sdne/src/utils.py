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

def get_similarity(result):
    """
    calculate similarity array

    Args:
        result(array): embedding array

    Returns:
        similarity array
    """
    print("getting similarity...")

    return np.dot(result, result.T)

def check_reconstruction(embeddings, graph, idx2node=None, k_query=None):
    """
    check reconstruction precision

    Args:
        embeddings(array): embedding array
        graph(DiGraph): original graph data
        idx2node(dict): the table for converting index to node
        k_query(list): k query set

    Returns:
        precision@k list
    """
    if k_query is None:
        k_query = [1, 10, 100, 1000, 10000]

    def get_precisionK(embeddings, graph, max_index, idx2node):
        similarity = get_similarity(embeddings).reshape(-1)
        sorted_idx = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sorted_idx = sorted_idx[::-1]
        for idx in sorted_idx:
            x = idx // graph.number_of_nodes()
            x = idx2node[x]
            y = idx % graph.number_of_nodes()
            y = idx2node[y]
            count += 1
            if (graph.has_edge(x, y) or graph.has_edge(y, x) or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break

        return precisionK

    print('\nReconstruction Precision K ', k_query)
    precisionK = get_precisionK(embeddings, graph, np.max(k_query), idx2node)
    ret = []
    for index in k_query:
        print(f"Precision@K({index})=\t{precisionK[index - 1]}")
        ret.append(precisionK[index - 1])

    return ret
