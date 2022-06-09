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


import argparse
import numpy as np
import pandas as pd
from mindspore import load_checkpoint
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--res_path", type=str, default="./python_SGCN/res", help="result numpy path")
    parser.add_argument("--res_path2", type=str, default="res", help="result numpy path")
    parser.add_argument("--data_dir", type=str, default="../data/bitcoin_alpha.csv",
                        help="Dataset")
    parser.add_argument("--ckpt_path", type=str,
                        default="../data/sgcn_ascend_v130_bitcoinalpha_research_gnn_bs64_AUC80.81.ckpt",
                        help="ckpt")
    parser.add_argument("--dataset_type", type=str, default="alpha", help="result numpy path")
    args_opt = parser.parse_args()
    return args_opt

#  load static result
def read_restxt(res_path):
    return np.loadtxt(res_path)


def read_graph(data_path):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    Args:
        args(Arguments): Arguments object.

    Returns:
        edge(dicts): Edges dictionary.
    """
    dataset = pd.read_csv(data_path).values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges

def setup_dataset(data_path):
    edges = read_graph(data_path)
    positive_edges, test_positive_edges = train_test_split(edges["positive_edges"], test_size=0.2, random_state=1)
    negative_edges, test_negative_edges = train_test_split(edges["negative_edges"], test_size=0.2, random_state=1)
    ecount = len(positive_edges + negative_edges)
    positive_edges = np.array(positive_edges, dtype=np.int32).T
    negative_edges = np.array(negative_edges, dtype=np.int32).T
    y = np.array([0 if i < int(ecount / 2) else 1 for i in range(ecount)] + [2] * (ecount * 2))
    y = np.array(y, np.int32)
    print('self.positive_edges', positive_edges.shape, type(positive_edges))
    print('self.negative_edges', negative_edges.shape, type(negative_edges))
    print('self.y', y.shape, type(y))
    print(positive_edges.dtype, negative_edges.dtype, y.dtype)
    return test_positive_edges, test_negative_edges

def remove_self_loops(edge_index):
    """
    remove self loops
    Args:
        edge_index (LongTensor): The edge indices.

    Returns:
        Tensor(edge_index): removed self loops
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return Tensor(edge_index)

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

def softmax(x):
    """Softmax"""
    t_max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - t_max)  # subtracts each row with its max value
    t_sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / t_sum
    return f_x

def test_result(arg):
    pos_test, neg_test = setup_dataset(arg.data_dir)
    preds = read_restxt(arg.res_path)
    if arg.dataset_type == "otc":
        preds = preds.reshape(5881, 64)
    else:
        preds = preds.reshape(3783, 64)
    param_dict = load_checkpoint(args.ckpt_path)
    weights = np.array(param_dict['regression_weights'].asnumpy())
    bias = np.array(param_dict['regression_bias'].asnumpy())
    score_positive_edges = np.array(pos_test, dtype=np.int32).T
    score_negative_edges = np.array(neg_test, dtype=np.int32).T
    test_positive_z = np.concatenate((preds[score_positive_edges[0, :], :],
                                      preds[score_positive_edges[1, :], :]), axis=1)
    test_negative_z = np.concatenate((preds[score_negative_edges[0, :], :],
                                      preds[score_negative_edges[1, :], :]), axis=1)
    scores = np.dot(np.concatenate((test_positive_z, test_negative_z), axis=0), weights) + bias
    probability_scores = np.exp(softmax(scores))
    predictions = probability_scores[:, 0] / probability_scores[:, 0: 2].sum(1)
    targets = [0] * len(pos_test) + [1] * len(neg_test)
    auc, f1 = calculate_auc(targets, predictions)
    print("Test set results:", "auc=", "{:.5f}".format(auc), "f1=", "{:.5f}".format(f1))

if __name__ == '__main__':
    args = parse_args()
    test_result(args)
