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
Evaluation script
"""
import argparse
import os

import numpy as np
from mindspore import context
from mindspore.common import set_seed
from sklearn.model_selection import train_test_split

from src.ms_utils import read_graph
from src.ms_utils import setup_features


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

    X = setup_features(args, positive_edges, negative_edges, edges["ncount"])
    X = np.array(X.tolist())
    positive_edges = np.array(positive_edges, dtype=np.int32).T
    negative_edges = np.array(negative_edges, dtype=np.int32).T
    y = np.array([0 if i < int(ecount / 2) else 1 for i in range(ecount)] + [2] * (ecount * 2))
    y = np.array(y, np.int32)
    X = np.array(X, np.float32)
    print(positive_edges.dtype, negative_edges.dtype, y.dtype)
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
    """main"""
    # Set DEVICE_ID
    parser = argparse.ArgumentParser(description="SGCN eval")
    parser.add_argument("--device_id", help="device_id", default=2, type=int)
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend"], help="device target (default: Ascend)")
    parser.add_argument("--edge_path", nargs="?",
                        default="./input/bitcoin_alpha.csv", help="Edge list csv.")
    parser.add_argument("--result_path", nargs="?",
                        default="./ascend310_infer/input/", help="result path")
    parser.add_argument("--features-path", nargs="?",
                        default="./input/bitcoin_alpha.csv", help="Edge list csv.")
    parser.add_argument("--test-size", type=float,
                        default=0.2, help="Test dataset size. Default is 0.2.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")
    parser.add_argument("--spectral-features", default=True, dest="spectral_features", action="store_true")
    parser.add_argument("--reduction-iterations", type=int,
                        default=30, help="Number of SVD iterations. Default is 30.")
    parser.add_argument("--reduction-dimensions", type=int,
                        default=64, help="Number of SVD feature extraction dimensions. Default is 64.")

    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    set_seed(args.seed)
    edges = read_graph(args)
    input_x, pos_edg, neg_edg, pos_test, neg_test = setup_dataset(args, edges)
    repos, reneg = remove_self_loops(pos_edg), remove_self_loops(neg_edg)
    repos.tofile(os.path.join(args.result_path, '00_data', 'repos.bin'))
    reneg.tofile(os.path.join(args.result_path, '01_data', 'reneg.bin'))
    np.save(os.path.join(args.result_path, 'pos_test.npy'), pos_test)
    np.save(os.path.join(args.result_path, 'neg_test.npy'), neg_test)
    np.save(os.path.join(args.result_path, 'input_x.npy'), input_x)


if __name__ == "__main__":
    main()
