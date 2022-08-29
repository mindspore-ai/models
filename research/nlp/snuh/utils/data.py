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
"""Data utils."""
from copy import deepcopy
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds
import scipy.io
from scipy.special import softmax
transpose = ops.Transpose()
matmul = ops.MatMul()
mind_topk = ops.TopK(sorted=True)
cast = ops.Cast()

class Data:
    """dataset class"""
    def __init__(self, file_path, num_neighbors):
        self.file_path = file_path
        self.load_datasets()
        self.get_topk_using_cosine_sim(topk=num_neighbors, query_batch_size=500, doc_batch_size=100, use_test=False)

    def load_datasets(self):
        raise NotImplementedError

    def get_topk_using_cosine_sim(self, topk, query_batch_size, doc_batch_size, use_test=False):
        raise NotImplementedError

    def get_loaders(self, num_trees, alpha, batch_size):
        raise NotImplementedError

class LabeledDocuments(Data):
    """labeled data class"""
    def __init__(self, file_path, num_neighbors):
        super().__init__(file_path=file_path, num_neighbors=num_neighbors)

    def load_datasets(self):
        dataset = scipy.io.loadmat(self.file_path)

        # (num documents) x (vocab size) tensors containing tf-idf values
        self.x_train = dataset['train'].toarray()
        self.x_val = dataset['cv'].toarray()
        self.x_test = dataset['test'].toarray()

        # (num documents) x (num labels) tensors containing {0,1}
        self.y_train = dataset['gnd_train']
        self.y_val = dataset['gnd_cv']
        self.y_test = dataset['gnd_test']

        self.vocab_size = self.x_train.shape[1]
        self.num_labels = self.y_train.shape[1]
        print("train num:", self.x_train.shape[0], "val num:", self.x_val.shape[0], "test num:",\
            self.x_test.shape[0], "vocab size:", self.vocab_size)

    def get_loaders(self, num_trees, alpha, batch_size):
        self.edges = self.get_spanning_trees(num_trees, alpha)
        self.num_nodes = self.x_train.shape[0]
        self.num_edges = self.edges.shape[0]

        train_dataset = ds.GeneratorDataset(source=TrainGenerator(self.x_train, self.y_train, self.edges),\
            column_names=["data", "label", "edges1", "edges2", "weight"], shuffle=True)
        database_dataset = ds.GeneratorDataset(source=TestGenerator(self.x_train, self.y_train),\
            column_names=["data", "label"], shuffle=False)
        val_dataset = ds.GeneratorDataset(source=TestGenerator(self.x_val, self.y_val),\
            column_names=["data", "label"], shuffle=False)
        test_dataset = ds.GeneratorDataset(source=TestGenerator(self.x_test, self.y_test),\
            column_names=["data", "label"], shuffle=False)

        train_loader = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
        database_loader = database_dataset.batch(batch_size=batch_size, drop_remainder=True)
        val_loader = val_dataset.batch(batch_size=len(self.x_val), drop_remainder=True)
        test_loader = test_dataset.batch(batch_size=len(self.x_test), drop_remainder=True)
        return train_loader, database_loader, val_loader, test_loader

    def get_topk_using_cosine_sim(self, topk, query_batch_size, doc_batch_size, use_test=False):
        documents = deepcopy(self.x_train)
        queries = deepcopy(self.x_test) if use_test else deepcopy(self.x_train)
        y_documents = deepcopy(self.y_train)
        y_queries = deepcopy(self.y_test) if use_test else deepcopy(self.y_train)

        # normalize
        documents = documents / np.linalg.norm(documents, axis=-1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=-1, keepdims=True)

        # compute cosine similarity
        cos_sim_scores = Tensor(np.dot(queries, documents.T), ms.float32)

        if use_test: topk = 100
        scores, indices = mind_topk(cos_sim_scores, topk+1)
        self.topk_scores = scores[:, 1: ].asnumpy()
        self.topk_indices = indices[:, 1: ].asnumpy()

        # test
        if use_test:
            print("test Top100 accuracy: {:.4f}".format(np.mean(np.sum(np.repeat(np.expand_dims(y_queries, axis=1),\
                topk, axis=1) * y_documents[self.topk_indices], axis=-1) > 0)))
            exit()
        else:
            print("graph (K={:d}) accuracy: {:.4f}".format(topk,\
                np.mean(np.sum(np.repeat(np.expand_dims(y_queries, axis=1), topk, axis=1) *\
                    y_documents[self.topk_indices], axis=-1) > 0)))

        del documents, queries, y_documents, y_queries

    def get_spanning_trees(self, num_trees, alpha):
        """generate spanning tree"""
        edges = self.topk_indices
        edges_scores = softmax(self.topk_scores / alpha, axis=-1)

        n = edges.shape[0]
        w_m = {}
        for _ in range(num_trees):
            visited = np.array([False for i in range(n)])
            while False in visited:
                init_node = np.random.choice(np.where(np.equal(visited, False))[0], 1)[0]   # np.where(visited == False)[0]
                visited[init_node] = True
                queue = [init_node]
                while queue:
                    now = queue[0]
                    visited[now] = True
                    edge_idx = np.where(np.equal(visited[edges[now]], False))[0]   # np.where(visited[edges[now]] == False)[0]
                    if not list(edge_idx):   # len(edge_idx) == 0
                        queue.pop(-1)
                        break
                    next_ = np.random.choice(edges[now][edge_idx], 1, p=edges_scores[now][edge_idx]\
                        / np.sum(edges_scores[now][edge_idx]))[0]
                    visited[next_] = True
                    queue.append(next_)
                    if (now * n + next_) not in w_m:
                        w_m[now * n + next_] = 1
                    else:
                        w_m[now * n + next_] += 1

        edges = [[key // n, key % n, val / num_trees] for key, val in w_m.items()]
        np.random.shuffle(edges)
        return np.array(edges)


class TrainGenerator():
    """training data generator"""
    def __init__(self, data, labels, edges):
        self.data = data
        self.labels = labels
        self.edges = edges

        self.edge_idx = 0

    def __getitem__(self, index):
        if self.edge_idx >= len(self.edges):
            self.edge_idx = 0
        text = self.data[index]
        labels = self.labels[index]
        edge1 = self.data[int(self.edges[self.edge_idx][0])]
        edge2 = self.data[int(self.edges[self.edge_idx][1])]
        weight = self.edges[self.edge_idx][2]
        self.edge_idx += 1
        return text.astype(np.float32), labels.astype(np.float32), edge1.astype(np.float32),\
                edge2.astype(np.float32), weight.astype(np.float32)

    def __len__(self):
        return len(self.data)

class TestGenerator():
    """testing data generator"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.float32)

    def __len__(self):
        return len(self.data)
