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
Export script
"""
import argparse
import ast

from mindspore import Tensor
from mindspore import context
from mindspore import export
from mindspore import load_checkpoint
from mindspore import load_param_into_net

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
    parser.add_argument("--checkpoint_file", type=str, default='sgcn_otc_auc.ckpt', help="Checkpoint file path.")
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
    parser.add_argument("--file_name", type=str, default="sgcn", help="output file name.")
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
    parser.add_argument("--norm", type=ast.literal_eval, default=True,
                        help="If true scatter_mean is used, else scatter_add.")
    parser.add_argument("--norm-embed", type=ast.literal_eval, default=True, help="Normalize embedding or not.")
    parser.add_argument("--bias", type=ast.literal_eval, default=True, help="Add bias or not.")

    args = parser.parse_args()

    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    # Create network
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    dataset = trainer.setup_dataset()
    input_x, pos_edg, neg_edg = dataset[0], dataset[1], dataset[2]
    repos, reneg = remove_self_loops(pos_edg), remove_self_loops(neg_edg)
    net = SignedGraphConvolutionalNetwork(input_x, args.norm, args.norm_embed, args.bias)
    # Load parameters from checkpoint into network
    param_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(net, param_dict)
    # export
    export(net, repos, reneg,
           file_name=args.file_name, file_format=args.file_format)
    print("==========================================")
    print(args.file_name + ".mindir exported successfully!")
    print("==========================================")


if __name__ == "__main__":
    main()
