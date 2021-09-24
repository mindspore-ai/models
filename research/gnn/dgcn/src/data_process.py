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
# coding=utf-8
"""
create adjacency matrix, node features, labels, and mask for training.
Cora ,Citeseer and Pubmed datasets are supported by our example
"""
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_graph_data(DATASET='cora'):
    """Get adjacency_matrix,labels and features matrix from the dataset"""
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(pickle.load(open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb'), encoding='iso-8859-1'))
    x, y, tx, ty, allx, ally, graph = tuple(OBJECTS)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(DATASET))
    # the indices of test instances in graph
    test_idx_range = np.sort(test_idx_reorder)

    if DATASET == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # get the features:X shape(3327,3703)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print("Feature matrix:" + str(features.shape))

    # get the labels: y shape(3327,6)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    print("Label matrix:" + str(labels.shape))

    # get the adjacent matrix: A shape(3327,3327)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print("Adjacent matrix:" + str(adj.shape))

    # test, validation, train
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels
    