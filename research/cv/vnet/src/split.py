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
"""create csv file to train."""

import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='create csv file to train')
parser.add_argument('--out_path', type=str, default="./", help="Path of csv files, default is ./")
parser.add_argument('--random', type=int, default=0, help='random to create csv files, 0 -- not random, 1 -- random')


def split_train_eval(out_path, max_len=50, index_len=10, random=False, seed=0):
    """split train and eval sets"""
    np.random.seed(seed)
    number_list = np.arange(max_len)
    eval_list = np.random.choice(number_list, size=index_len, replace=False)
    if not random:
        eval_list = np.array([11, 15, 19, 30, 42, 43, 44, 46, 48, 49])
    train_name_list = ['Case' + str(k).zfill(2) + '.mhd' for k in number_list if k not in eval_list]
    eval_name_list = ['Case' + str(k).zfill(2) + '.mhd' for k in eval_list]
    train_name_p = pd.DataFrame(train_name_list)
    train_name_p.to_csv(os.path.join(out_path, 'train.csv'), header=None, index=False)
    eval_name_p = pd.DataFrame(eval_name_list)
    eval_name_p.to_csv(os.path.join(out_path, 'val.csv'), header=None, index=False)


def main():
    """create csv file to train"""
    args = parser.parse_args()
    split_train_eval(out_path=args.out_path, random=args.random)


if __name__ == '__main__':
    main()
