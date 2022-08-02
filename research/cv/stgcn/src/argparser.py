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
"""Command line arguments parsing"""

import argparse
import ast


def arg_parser():
    """Parsing of command line arguments"""
    parser = argparse.ArgumentParser('stgcn parameters')

    # The way of training
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--epochs', type=int, default=500, help='Epochs to train model')
    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run on modelarts')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--device_id', type=int, default=0, help='Device id, default is 0.')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Whether to save checkpoint')

    # Path to data and checkpoints
    parser.add_argument('--data_url', type=str, required=True, help='Dataset directory.')
    parser.add_argument('--train_url', type=str, required=True, help='Save checkpoint directory.')
    parser.add_argument('--data_path', type=str, default="vel.csv", help='Dataset file of vel.')
    parser.add_argument('--wam_path', type=str, default="adj_mat.csv", help='Dataset file of warm.')
    parser.add_argument('--label_dir', type=str, default='', help='label data directory.')
    parser.add_argument('--ckpt_url', type=str, default="", help='Path to saved checkpoint.')
    parser.add_argument("--file_name", type=str, default="stgcn", help="output file name.")
    parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR",
                        help="file format to export")
    parser.add_argument('--result_dir', type=str, default="./result_Files", help='infer result dir.')

    # Parameters for training
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument('--n_pred', type=int, default=3, help='The number of time interval for predcition, default: 3')
    parser.add_argument('--opt', type=str, default='RMSProp', help='optimizer, default as AdamW')

    # network
    parser.add_argument('--graph_conv_type', type=str, default="gcnconv", choices=["gcnconv", "chebconv"],
                        help='Graph convolution type, default: gcnconv')

    return parser.parse_args()
