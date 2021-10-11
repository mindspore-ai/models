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
"""config"""
import argparse

def parse_opts():
    """arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size(default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--natt', type=int, default=8, metavar='N', help='natt')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='LR', help='weight decay (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--data-dir', type=str, default='', help='dataset file path')
    parser.add_argument('--result-dir', type=str, default='', help='result saved file path')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint file path')
    parser.add_argument('--fc', type=int, default=1, metavar='N', help='fc')
    parser.add_argument('--device-id', type=int, default=0, metavar='N', help='device-id')
    parser.add_argument('--distributed', type=int, default=0, metavar='N', help='distributed')

    args = parser.parse_args()
    return args
