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
Evaluation script
"""
import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    Args:
        args(Arguments): Arguments object.

    Returns:
        edge(dicts): Edges dictionary.
    """
    dataset = pd.read_csv(args.features_path).values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges


def setup_dataset(args, edges):
    """
    Returns:
        X(Tensor): Dataset.
        positive_edges(Tensor): Positive edges for training.
        negative_edges(Tensor): Negative edges for training.
        test_positive_edges(Tensor): Positive edges for testing.
        test_negative_edges(Tensor): Negative edges for testing.
    """
    positive_edges, test_positive_edges = train_test_split(edges["positive_edges"],
                                                           test_size=args.test_size, random_state=1)

    negative_edges, test_negative_edges = train_test_split(edges["negative_edges"],
                                                           test_size=args.test_size, random_state=1)
    ecount = len(positive_edges + negative_edges)

    X = np.array(pd.read_csv(args.features_path))
    X = np.array(X.tolist())
    positive_edges = np.array(positive_edges, dtype=np.int32).T
    negative_edges = np.array(negative_edges, dtype=np.int32).T
    y = np.array([0 if i < int(ecount / 2) else 1 for i in range(ecount)] + [2] * (ecount * 2))
    y = np.array(y, np.int32)
    X = np.array(X, np.float32)
    return X, positive_edges, negative_edges, test_positive_edges, test_negative_edges


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
    return edge_index


def main():
    def w2txt(file, data):
        s = ""
        for i in range(len(data[0])):
            s = s + str(data[0][i])
            s = s + " "
        with open(file, "w") as f:
            f.write(s)

    # Set DEVICE_ID
    parser = argparse.ArgumentParser(description="SGCN eval")
    parser.add_argument("--device_id", help="device_id", default=2, type=int)
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend"], help="device target (default: Ascend)")
    parser.add_argument("--edge_path", nargs="?",
                        default="./data/bitcoin_alpha.csv", help="Edge list csv.")
    parser.add_argument("--result_path", nargs="?",
                        default="./data/", help="result path")
    parser.add_argument("--features_path", nargs="?",
                        default="./data/bitcoin_alpha.csv", help="Edge list csv.")
    parser.add_argument("--test-size", type=float,
                        default=0.2, help="Test dataset size. Default is 0.2.")
    parser.add_argument("--dataset_type", type=str, default="alpha")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")
    parser.add_argument("--spectral-features", default=True, dest="spectral_features", action="store_true")
    parser.add_argument("--reduction-iterations", type=int,
                        default=30, help="Number of SVD iterations. Default is 30.")
    parser.add_argument("--reduction-dimensions", type=int,
                        default=64, help="Number of SVD feature extraction dimensions. Default is 64.")

    args = parser.parse_args()
    edges = read_graph(args)
    dataset = setup_dataset(args, edges)
    pos_edg, neg_edg = dataset[1], dataset[2]
    repos, reneg = remove_self_loops(pos_edg), remove_self_loops(neg_edg)

    if args.dataset_type == "alpha":
        repos = np.array(repos, dtype=np.int32).reshape(1, 20430)
        reneg = np.array(reneg, dtype=np.int32).reshape(1, 2098)
    else:
        repos = np.array(repos, dtype=np.int32).reshape(1, 29248)
        reneg = np.array(reneg, dtype=np.int32).reshape(1, 5044)

    w2txt(os.path.join(args.result_path, "repos.txt"), repos)
    w2txt(os.path.join(args.result_path, "reneg.txt"), reneg)

if __name__ == "__main__":
    main()
