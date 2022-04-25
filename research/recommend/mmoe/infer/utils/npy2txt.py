# Copyright (c) 2022. Huawei Technologies Co., Ltd
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

import os
import argparse
import numpy as np

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="prepare txt")
    parser.add_argument('--data_dir', type=str, default='../data/input/')
    parser.add_argument('--data_file', type=str, default='data_{}.npy')
    parser.add_argument('--income_file', type=str, default='income_labels_{}.npy')
    parser.add_argument('--married_file', type=str, default='married_labels_{}.npy')
    parser.add_argument('--mode', type=str, default='eval')
    args_opt = parser.parse_args()
    return args_opt

def run():
    """prepare txt data"""
    args = parse_args()
    # load npy data
    data = np.load(os.path.join(args.data_dir, args.data_file.format(args.mode)))
    income = np.load(os.path.join(args.data_dir, args.income_file.format(args.mode)))
    married = np.load(os.path.join(args.data_dir, args.married_file.format(args.mode)))

    np.savetxt(os.path.join(args.data_dir, args.data_file.split('.')[0].format(args.mode)+'.txt'), data, delimiter='\t')
    np.savetxt(os.path.join(args.data_dir, args.income_file.split('.')[0].format(args.mode)+'.txt'), \
                income, delimiter='\t')
    np.savetxt(os.path.join(args.data_dir, args.married_file.split('.')[0].format(args.mode)+'.txt'), \
                married, delimiter='\t')

if __name__ == '__main__':
    run()
