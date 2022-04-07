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
python dataset.py
"""
import networkx as nx
import numpy as np
import scipy.sparse as sp

import mindspore
import mindspore.dataset as ds

from .utils import preprocess_nxgraph

class MyAccessible:
    """
    MyAccessible

    Args:
        dataset_name(str): dataset name
        dataset_file(str): dataset file path
        batch(int): batch size
        delimiter(str): a delimiter in dataset file
        linkpred(dict): link prediction config

    Returns:
        accessor(MyAccessible): a dataset accessor
    """
    def __init__(self, dataset_name, dataset_file, batch=32, delimiter=','):
        self.dataset_name = dataset_name
        self.batch = batch
        self.graph = nx.read_edgelist(dataset_file, create_using=nx.DiGraph(),
                                      delimiter=delimiter, data=[('weight', float)], nodetype=int)
        self.node_size = self.graph.number_of_nodes()
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)
        print("number of nodes: ", self.node_size)
        self.A = None
        self.L = None
        if dataset_name == 'WIKI':
            self.A, self.L = self._create_A_L(self.graph, self.node2idx)
        else:
            self.A = self._create_A(self.graph, self.node2idx)
        self.index = 0
        self.order = np.arange(self.node_size)
        self.max_batch = self.node_size // self.batch
        self.max_batch += 0 if self.node_size % self.batch == 0 else 1

    def __getitem__(self, index):
        """
        getitem
        """
        if self.index >= self.max_batch:
            self.index = 0
            if self.dataset_name == 'WIKI':
                np.random.shuffle(self.order)

        st = self.index * self.batch
        en = min((self.index + 1) * self.batch, self.node_size)
        ind = self.order[st : en]
        self.index = self.index + 1

        if self.dataset_name == 'WIKI':
            return self.A[ind, :].todense().A, self.L[ind, :][:, ind].todense().A

        return self.A[ind, :].todense().A, self.A[ind, :][:, ind].todense().A

    def __len__(self):
        """
        max batch number
        """
        return self.max_batch

    def _create_A(self, graph, node2idx):
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
            A_data.append(edge_weight)
            A_row_index.append(node2idx[v2])
            A_col_index.append(node2idx[v1])

        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size), dtype=np.float32)

        return A

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

        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size), dtype=np.float32)
        A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                           shape=(node_size, node_size), dtype=np.float32)

        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0], dtype=A_.dtype)
        L = D - A_
        return A, L

class GraphDataset:
    """
    GraphDataset

    Args:
        dataset_name(str): dataset name
        dataset_file(str): dataset file path
        batch(int): batch size
        delimiter(str): a delimiter in dataset file
        linkpred(dict): link prediction config

    Returns:
        Wrapper(GraphDataset): MyAccessible's Wrapper
    """
    def __init__(self, dataset_name, dataset_file, batch=64, delimiter=','):
        self.dataset_name = dataset_name
        self.dataset = MyAccessible(dataset_name, dataset_file, batch=batch, delimiter=delimiter)

    def get_data(self, frac=1.0, use_rand=False):
        """
        data sampler
        """
        sz = int(self.dataset.node_size * frac)
        if use_rand:
            index = np.random.permutation(self.dataset.node_size)
            index = index[:sz]
            index = np.sort(index)
            return index, mindspore.Tensor(self.dataset.A[index, :].todense().A)

        return np.arange(sz), mindspore.Tensor(self.dataset.A[:sz, :].todense().A)

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

    def get_data_size(self):
        """
        data size
        """
        return len(self.dataset)

    def get_nodes(self):
        """
        nodes list
        """
        return self.dataset.graph.nodes()

    def get_dataset(self):
        """
        a dataset generator
        """
        if self.dataset_name == 'WIKI':
            return ds.GeneratorDataset(source=self.dataset, column_names=["X", "L"],
                                       column_types=[mindspore.float32, mindspore.float32])

        return ds.GeneratorDataset(source=self.dataset, column_names=['X', 'Xadj'],
                                   column_types=[mindspore.float32, mindspore.float32])
