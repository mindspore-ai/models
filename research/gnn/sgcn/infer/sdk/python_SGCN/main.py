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
import time
import numpy as np
import pandas as pd
from SdkApi import SdkApi
from sklearn.model_selection import train_test_split

STREAM_NAME = b'sgcn'
TENSOR_DTYPE_FLOAT16 = 1
TENSOR_DTYPE_INT32 = 3


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert process")
    parser.add_argument("--pipeline_path", type=str, default="../pipeline/sgcn.pipeline", help="SDK infer pipeline")
    parser.add_argument("--data_dir", type=str,
                        default="../../data/bitcoin_alpha.csv", help="Path where the dataset is saved")
    args_opt = parser.parse_args()
    return args_opt


def inference():
    args = parse_args()

    # init stream manager
    sdk_api = SdkApi(args.pipeline_path)
    if not sdk_api.init():
        exit(-1)

    start_time = time.time()
    dataset = setup_dataset(args.data_dir)
    pos_edg, neg_edg = dataset[1], dataset[2]
    repos, reneg = remove_self_loops(pos_edg), remove_self_loops(neg_edg)
    sdk_api.send_tensor_input(STREAM_NAME, 0, b'appsrc0', repos, repos.shape, TENSOR_DTYPE_INT32)
    sdk_api.send_tensor_input(STREAM_NAME, 1, b'appsrc1', reneg, reneg.shape, TENSOR_DTYPE_INT32)
    print("Getting result")
    result = sdk_api.get_result(STREAM_NAME)
    pred = np.array(
        [np.frombuffer(result.tensorPackageVec[k].tensorVec[0].dataStr, dtype=np.float32) for k in range(2)])
    end_time = time.time() - start_time
    print(f"The inference time is {end_time}")
    np.savetxt('res', pred, fmt="%f")

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
    """
    Returns:
        X(Tensor): Dataset.
        positive_edges(Tensor): Positive edges for training.
        negative_edges(Tensor): Negative edges for training.
        test_positive_edges(Tensor): Positive edges for testing.
        test_negative_edges(Tensor): Negative edges for testing.
    """
    edges = read_graph(data_path)
    positive_edges, test_positive_edges = train_test_split(edges["positive_edges"], test_size=0.2, random_state=1)
    negative_edges, test_negative_edges = train_test_split(edges["negative_edges"], test_size=0.2, random_state=1)
    ecount = len(positive_edges + negative_edges)
    X = np.array(pd.read_csv(data_path))
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

if __name__ == '__main__':
    inference()
