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

"""
sample script of mmoe calculating metric
"""

import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description='calc metric')
    parser.add_argument('--data_dir', type=str, default='../output')
    parser.add_argument('--income_preds', type=str, default='income_preds_{}.npy')
    parser.add_argument('--married_preds', type=str, default='married_preds_{}.npy')
    parser.add_argument('--income_labels', type=str, default='income_labels_{}.npy')
    parser.add_argument('--married_labels', type=str, default='married_labels_{}.npy')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--metric_file', type=str, default='./metric.txt')
    args_opt = parser.parse_args()
    return args_opt

def get_auc(labels, preds):
    return roc_auc_score(labels, preds)

def run():
    """calc metric"""
    args = parse_args()
    income_preds = np.load(os.path.join(args.data_dir, args.income_preds.format(args.mode)))
    income_preds = income_preds.flatten().tolist()
    married_preds = np.load(os.path.join(args.data_dir, args.married_preds.format(args.mode)))
    married_preds = married_preds.flatten().tolist()

    income_labels = np.load(os.path.join(args.data_dir, args.income_labels.format(args.mode)))
    income_labels = income_labels.flatten().tolist()
    married_labels = np.load(os.path.join(args.data_dir, args.married_labels.format(args.mode)))
    married_labels = married_labels.flatten().tolist()

    income_auc = get_auc(income_labels, income_preds)
    married_auc = get_auc(married_labels, married_preds)
    print('<<========  Infer Metric ========>>')
    print('Mode: {}'.format(args.mode))
    print('Income auc: {}'.format(income_auc))
    print('Married auc: {}'.format(married_auc))
    print('<<===============================>>')
    fo = open(args.metric_file, "w")
    fo.write('Mode: {}\n'.format(args.mode))
    fo.write('Income auc: {}\n'.format(income_auc))
    fo.write('Married auc: {}\n'.format(married_auc))
    fo.close()

if __name__ == '__main__':
    run()
