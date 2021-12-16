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
"""Command line arguments parsing"""

import argparse
import ast


def arg_parser():
    """Parsing of command line arguments"""
    parser = argparse.ArgumentParser(description="Image classification")
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help="Device target, support Ascend and GPU.")
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    parser.add_argument('--device_num', type=int, default=1, help='Device number')
    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run mode')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    # parser.add_argument('--config_path', type=str, default=None, help='Path to config')

    return parser.parse_args()
