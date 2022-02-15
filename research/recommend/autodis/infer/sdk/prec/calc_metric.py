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
sample script of autodis calculating metric
"""

import argparse
import numpy as np

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description='calc metric')
    parser.add_argument('--result_file', type=str, default='../output/result.txt')
    parser.add_argument('--output_file', type=str, default='./metric.txt')
    args_opt = parser.parse_args()
    return args_opt

def get_acc(labels, preds):
    """Accuracy"""
    accuracy = np.sum(labels == preds) / len(labels)
    return accuracy

def get_auc(labels, preds, n_bins=10000):
    """ROC_AUC"""
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    if total_case == 0:
        return 0
    pos_histogram = [0 for _ in range(n_bins+1)]
    neg_histogram = [0 for _ in range(n_bins+1)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins+1):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)

def run():
    """
    calc metric
    """
    args = parse_args()
    data = np.loadtxt(args.result_file, delimiter="\t", skiprows=1)
    data.shape = -1, 3
    acc = get_acc(data[:, 0], data[:, 2])
    auc = get_auc(data[:, 0], data[:, 1])
    fo = open(args.output_file, "w")
    fo.write("Infer acc:{}\nInfer auc:{}".format(acc, auc))
    fo.close()
    print("Infer acc:{}\nInfer auc:{}".format(acc, auc))

if __name__ == '__main__':
    run()
