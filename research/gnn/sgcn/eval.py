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
import ast

import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore.common import set_seed

from src.ms_utils import calculate_auc
from src.ms_utils import read_graph
from src.sgcn import SignedGCNTrainer
from src.sgcn import SignedGraphConvolutionalNetwork


def remove_self_loops(edge_index):
    """
    remove self loops
    Args:
        edge_index (LongTensor): The edge indices.

    Returns:
        Tensor(edge_index): removed self loops
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index.asnumpy()[:, mask.asnumpy()]
    return Tensor(edge_index)

def main():
    """main"""
    # Set DEVICE_ID
    parser = argparse.ArgumentParser(description="SGCN eval")
    parser.add_argument("--device_id", help="device_id", default=0, type=int)
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend", "GPU"], help="device target (default: Ascend)")
    parser.add_argument("--checkpoint_auc", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--checkpoint_f1", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--edge_path", nargs="?",
                        default="./input/bitcoin_otc.csv", help="Edge list csv.")
    parser.add_argument("--features-path", nargs="?",
                        default="./input/bitcoin_otc.csv", help="Edge list csv.")
    parser.add_argument("--test-size", type=float,
                        default=0.2, help="Test dataset size. Default is 0.2.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")
    parser.add_argument("--spectral-features", default=True, dest="spectral_features", action="store_true")
    parser.add_argument("--reduction-iterations", type=int,
                        default=30, help="Number of SVD iterations. Default is 30.")
    parser.add_argument("--reduction-dimensions", type=int,
                        default=64, help="Number of SVD feature extraction dimensions. Default is 64.")
    parser.add_argument("--norm", type=ast.literal_eval, default=True,
                        help="If true scatter_mean is used, else scatter_add.")
    parser.add_argument("--norm-embed", type=ast.literal_eval, default=True, help="Normalize embedding or not.")
    parser.add_argument("--bias", type=ast.literal_eval, default=True, help="Add bias or not.")

    args = parser.parse_args()
    set_seed(args.seed)
    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    # Create network
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    input_x, pos_edg, neg_edg, pos_test, neg_test = trainer.setup_dataset()
    repos, reneg = remove_self_loops(pos_edg), remove_self_loops(neg_edg)
    net_auc = SignedGraphConvolutionalNetwork(input_x, args.norm, args.norm_embed, args.bias)
    net_f1 = SignedGraphConvolutionalNetwork(input_x, args.norm, args.norm_embed, args.bias)
    # Load parameters from checkpoint into network
    param_dict = load_checkpoint(args.checkpoint_auc)
    load_param_into_net(net_auc, param_dict)
    param_dict = load_checkpoint(args.checkpoint_f1)
    load_param_into_net(net_f1, param_dict)
    # Evaluation auc
    res = net_auc(repos, reneg)
    score_positive_edges = mnp.array(pos_test, dtype=mnp.int32).T
    score_negative_edges = mnp.array(neg_test, dtype=mnp.int32).T
    test_positive_z = ops.Concat(axis=1)((res[score_positive_edges[0, :], :],
                                          res[score_positive_edges[1, :], :]))
    test_negative_z = ops.Concat(axis=1)((res[score_negative_edges[0, :], :],
                                          res[score_negative_edges[1, :], :]))
    scores = ops.matmul(ops.Concat(axis=0)((test_positive_z, test_negative_z)),
                        net_auc.regression_weights) + net_auc.regression_bias
    probability_scores = ops.Exp()(ops.Softmax(axis=1)(scores))
    predictions = probability_scores[:, 0] / probability_scores[:, 0:2].sum(1)
    predictions = predictions.asnumpy()
    targets = [0] * len(pos_test) + [1] * len(neg_test)
    auc, _ = calculate_auc(targets, predictions)
    # Evaluation f1
    res = net_f1(repos, reneg)
    test_positive_z = ops.Concat(axis=1)((res[score_positive_edges[0, :], :],
                                          res[score_positive_edges[1, :], :]))
    test_negative_z = ops.Concat(axis=1)((res[score_negative_edges[0, :], :],
                                          res[score_negative_edges[1, :], :]))
    scores = ops.matmul(ops.Concat(axis=0)((test_positive_z, test_negative_z)),
                        net_f1.regression_weights) + net_f1.regression_bias
    probability_scores = ops.Exp()(ops.Softmax(axis=1)(scores))
    predictions = probability_scores[:, 0] / probability_scores[:, 0:2].sum(1)
    predictions = predictions.asnumpy()
    targets = [0] * len(pos_test) + [1] * len(neg_test)
    _, f1 = calculate_auc(targets, predictions)

    print("=====Evaluation Results=====")
    print('AUC:', '{:.6f}'.format(auc))
    print('F1-Score:', '{:.6f}'.format(f1))
    print("============================")


if __name__ == "__main__":
    main()
