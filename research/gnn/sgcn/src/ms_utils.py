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
"""Data reading utils."""
import numpy as np
import pandas as pd
from mindspore import Tensor
from mindspore import numpy as mnp
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from texttable import Texttable


def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    Args:
        args(Arguments): Arguments object.

    Returns:
        edge(dicts): Edges dictionary.
    """
    dataset = pd.read_csv(args.edge_path).values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def calculate_auc(targets, predictions):
    """
    Calculate performance measures on test dataset.
    Args:
        targets(Tensor): Ground truth.
        predictions(Tensor): Model outputs.

    Returns:
        auc(Float32): AUC result.
        f1(Float32): F1-Score result.
    """
    targets = [0 if target == 1 else 1 for target in targets]
    auc = roc_auc_score(targets, predictions)
    pred = [1 if p > 0.5 else 0 for p in predictions]
    f1 = f1_score(targets, pred)
    return auc, f1


def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 10 == 0])
    print(t.draw())


def setup_features(args, positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    Args:
        args(Arguments): Arguments object.
        positive_edges(List): Positive edges.
        negative_edges(List): Negative edges.
        node_count(Int): Number of nodes.

    Returns:
        X(Tensor): Dataset.
    """
    if args.spectral_features:
        X = create_spectral_features(args, positive_edges, negative_edges, node_count)
    else:
        X = create_general_features(args)
    return X


def create_general_features(args):
    """
    Reading features using the path.
    Args:
        args(Arguments): Arguments object.

    Returns:
        X(Tensor): Dataset.
    """
    X = mnp.array(pd.read_csv(args.features_path))
    return X


def create_spectral_features(args, positive_edges, negative_edges, node_count):
    """
    Creating spectral node features using the train dataset edges.
    Args:
        args(Arguments): Arguments object.
        positive_edges(List): Positive edges.
        negative_edges(List): Negative edges.
        node_count(Int): Number of nodes.

    Returns:
        X(Tensor): Dataset.
    """
    p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
    n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
    train_edges = p_edges + n_edges
    index_1 = [edge[0] for edge in train_edges]
    index_2 = [edge[1] for edge in train_edges]
    values = [1]*len(p_edges) + [-1]*len(n_edges)
    shaping = (node_count, node_count)
    signed_A = sparse.csr_matrix(
        sparse.coo_matrix(
            (values, (index_1, index_2)),
            shape=shaping,
            dtype=np.float32
        )
    )

    svd = TruncatedSVD(
        n_components=args.reduction_dimensions,
        n_iter=args.reduction_iterations,
        random_state=args.seed
    )
    svd.fit(signed_A)
    X = svd.components_.T
    return X


def maybe_num_nodes(edge_index, num_nodes=None):
    """
    Calculate the number of nodes
    Args:
        edge_index(Tensor): Indices of edges.
        num_nodes(Int): Number of nodes.

    Returns:
        res(Int): Max index of edges.
    """
    if num_nodes is not None:
        res = num_nodes
    elif isinstance(edge_index, Tensor):
        res = int(edge_index.max()) + 1
    else:
        if edge_index.size(0) > edge_index.size(1):
            res = edge_index.size(0)
        else:
            res = edge_index.size(1)
    return res
