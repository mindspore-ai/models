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
    parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-u", "--hidden_unit", type=int, default=10)
    parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help="Device target, support Ascend and GPU.")
    parser.add_argument("-di", "--device_id", type=int, default=0)
    parser.add_argument("--ckpt_dir", default='./checkpoint/', help='the path of output')
    parser.add_argument("--data_path", default='/data/omniglot_resized/',
                        help="Path where the dataset is saved")
    parser.add_argument("--data_url", default=None)
    parser.add_argument("--train_url", default=None)
    parser.add_argument("--cloud", default=None, help='if run on cloud')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--device_num', type=int, default=1, help='Device number')
    return parser.parse_args()
