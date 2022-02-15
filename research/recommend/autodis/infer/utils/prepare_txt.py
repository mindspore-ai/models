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
sample script of preparing txt for autodis infer
"""
import os
import argparse
import numpy as np
def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="prepare txt")
    parser.add_argument('--train_line_count', type=int, default=45840617)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--data_dir', type=str, default='../data/input/origin_data')
    parser.add_argument('--dst_dir', type=str, default='../data/input/origin_data')
    parser.add_argument('--data_input', type=str, default="train.txt")
    parser.add_argument('--data_output', type=str, default="test.txt")
    args, _ = parser.parse_known_args()
    return args

def run():
    """
    prepare txt data
    """
    args = parse_args()
    test_size = int(args.train_line_count * args.test_size)
    all_indices = [i for i in range(args.train_line_count)]
    np.random.seed(args.seed)
    np.random.shuffle(all_indices)
    print("all_indices.size:{}".format(len(all_indices)))
    test_indices_set = set(all_indices[:test_size])
    print("test_indices_set.size:{}".format(len(test_indices_set)))
    with open(os.path.join(args.data_dir, args.data_input), "r") as f:
        fo = open(os.path.join(args.dst_dir, args.data_output), "w")
        i = 0
        line = f.readline()
        while line:
            if i in test_indices_set:
                fo.write(line)
            i += 1
            line = f.readline()
        fo.close()

if __name__ == '__main__':
    run()
