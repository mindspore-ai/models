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
python dataset.py
"""
import random
import numpy
import networkx as nx
import scipy.sparse as sp

import mindspore
import mindspore.dataset as ds

from .utils import preprocess_nxgraph

class MyAccessible:
    """
    MyAccessible

    Args:
        dataset_file(str): dataset file path
        batch(int): batch size
        delimiter(str): a delimiter in dataset file
        linkpred(dict): link prediction config

    Returns:
        accessor(MyAccessible): a dataset accessor
    """
    def __init__(self, dataset_file, batch=32, delimiter=',', linkpred=None):
        self.batch = batch
        self.graph = nx.read_edgelist(dataset_file, create_using=nx.DiGraph(),
                                      delimiter=delimiter, data=[('weight', float)], nodetype=int)
        self.node_size = self.graph.number_of_nodes()
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)
        print("number of nodes: ", self.node_size)
        if linkpred is not None:
            remove_size = int(self.graph.number_of_edges() * (1 - linkpred['tr_frac']))
            remove_edges = random.sample(self.graph.edges(), remove_size)
            print("original number of edges: ", self.graph.number_of_edges())
            self.graph.remove_edges_from(remove_edges)
            print("cur number of edges: ", self.graph.number_of_edges())
            print("remove ", remove_size, " edges")
            if linkpred['store_tr_net']:
                data = []
                if linkpred['has_weight']:
                    data.append('weight')
                nx.write_edgelist(self.graph, './train_edges.txt', delimiter=delimiter, data=data)

        self.A, self.L = self._create_A_L(self.graph, self.node2idx)
        self.index = 0
        self.max_batch = self.node_size // self.batch
        self.max_batch += 0 if self.node_size % self.batch == 0 else 1

    def __getitem__(self, index):
        """
        getitem
        """
        ind1 = self.index * self.batch
        ind2 = min((self.index + 1) * self.batch, self.node_size)
        self.index = (self.index + 1) % self.max_batch
        return self.A[ind1:ind2, :].todense().A, self.L[ind1:ind2][:, ind1:ind2].todense().A

    def __len__(self):
        """
        max batch number
        """
        return self.max_batch

    def _create_A_L(self, graph, node2idx):
        """
        create Net matrix and Laplacian matrix
        """
        node_size = self.node_size
        A_data = []
        A_row_index = []
        A_col_index = []

        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get('weight', 1)

            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])

        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size), dtype=numpy.float32)
        A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                           shape=(node_size, node_size), dtype=numpy.float32)

        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0], dtype=A_.dtype)
        L = D - A_
        return A, L

class GraphDataset:
    """
    GraphDataset

    Args:
        dataset_file(str): dataset file path
        batch(int): batch size
        delimiter(str): a delimiter in dataset file
        linkpred(dict): link prediction config

    Returns:
        Wrapper(GraphDataset): MyAccessible's Wrapper
    """
    def __init__(self, dataset_file, batch=64, delimiter=',', linkpred=None):
        self.dataset = MyAccessible(dataset_file, batch=batch, delimiter=delimiter, linkpred=linkpred)

    def get_data(self, frac=1.0, use_rand=False):
        """
        data sampler
        """
        sz = int(self.dataset.node_size * frac)
        if use_rand:
            index = numpy.random.permutation(self.dataset.node_size)
            index = index[:sz]
            index = numpy.sort(index)
            return index, mindspore.Tensor(self.dataset.A[index, :].todense().A)

        return numpy.arange(sz), mindspore.Tensor(self.dataset.A[:sz, :].todense().A)

    def get_graph(self):
        """
        net graph
        """
        return self.dataset.graph

    def get_idx2node(self):
        """
        a table which convers index to node value
        """
        return self.dataset.idx2node

    def get_node2idx(self):
        """
        a table which convers node value to index
        """
        return self.dataset.node2idx

    def get_node_size(self):
        """
        node size
        """
        return self.dataset.node_size

    def get_nodes(self):
        """
        nodes list
        """
        return self.dataset.graph.nodes()

    def get_dataset(self):
        """
        a dataset generator
        """
        return ds.GeneratorDataset(source=self.dataset, column_names=["X", "L"],
                                   column_types=[mindspore.float32, mindspore.float32])
