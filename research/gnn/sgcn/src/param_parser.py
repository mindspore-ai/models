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
"""SGCN parameter parser."""
import argparse
import ast


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Bitcoin OTC dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run SGCN.")
    parser.add_argument("--distributed", type=ast.literal_eval, default=False, help="Distributed train")
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    parser.add_argument("--checkpoint_file", type=str, default='sgcn_alpha', help="Checkpoint file path.")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/bitcoin_alpha.csv", help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="./input/bitcoin_alpha.csv", help="Edge list csv.")

    parser.add_argument("--epochs",
                        type=int,
                        default=500, help="Number of training epochs. Default is 500.")

    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30, help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64, help="Number of SVD feature extraction dimensions. Default is 64.")

    parser.add_argument("--seed",
                        type=int,
                        default=42, help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--lamb",
                        type=float,
                        default=1.0, help="Embedding regularization parameter. Default is 1.0.")

    parser.add_argument("--test-size",
                        type=float,
                        default=0.2, help="Test dataset size. Default is 0.2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01, help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-5, help="Weight decay. Default is 10^-5.")

    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_true")

    parser.add_argument("--general-features",
                        dest="spectral_features",
                        action="store_false")

    parser.add_argument("--norm", type=ast.literal_eval, default=True, help="Normalize features or not.")
    parser.add_argument("--norm-embed", type=ast.literal_eval, default=True, help="Normalize embedding or not.")
    parser.add_argument("--bias", type=ast.literal_eval, default=True, help="Add bias or not.")

    parser.set_defaults(spectral_features=True)

    return parser.parse_args()
