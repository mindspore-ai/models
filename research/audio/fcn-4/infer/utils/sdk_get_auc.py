# coding:utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import numpy as np
import pandas as pd


def str2digit(s):
    if s.isdigit():
        return int(s)
    return s

def simplify_tagging_info(info_path="../data/config/", label_file="annotations_final.csv"):
    """ simplify_tagging_info """
    print("-"*25, "now in function simplify_tagging_info", "-"*25)
    T = []
    with open(os.path.join(info_path, label_file), 'rb') as info:
        data = info.readline()
        while data:
            T.append([str2digit(i[1:-1]) for i in data.strip().decode('utf-8').split("\t")])
            data = info.readline()
    annotation = pd.DataFrame(T[1:], columns=T[0])
    count = []
    for i in annotation.columns[1:-2]:
        count.append([annotation[i].sum() / len(annotation), i])
    count = sorted(count)
    full_label = []
    for i in count[-50:]:
        full_label.append(i[1])
    simplied_tag = []
    for i in T[1:]:
        index = [k for k, x in enumerate(i) if x == 1]
        label = [T[0][k] for k in index]
        L = [str(0) for _ in range(50)]
        L.append(i[-1])
        for j in label:
            if j in full_label:
                ind = full_label.index(j)
                L[ind] = '1'
        simplied_tag.append(L)
    txt_save_path = os.path.join(info_path, "music_tagging_tmp.txt")
    np.savetxt(txt_save_path, np.array(simplied_tag), fmt='%s', delimiter=',')
    csv_save_path = os.path.join(info_path, "music_tagging_tmp.csv")
    np.savetxt(csv_save_path, np.array(simplied_tag), fmt='%s', delimiter=',')
    print("successfully save tagging info in:\n", info_path)
    return simplied_tag

def get_labels(info_list, _result_path):
    """ get_labels """
    print("-"*25, "now in function get_labels", "-"*25)
    label_list = []
    pred_list = []
    print("info list length:\n", len(info_list))
    for label_info in info_list:
        file_name = label_info[-1][:-4] + ".txt"
        rst_file = os.path.join(_result_path, file_name)
        if os.path.exists(rst_file):
            true_label = np.array([str2digit(i) for i in label_info[:-1]])
            rst_data = np.loadtxt(rst_file, delimiter=',')
            label_list.append(true_label)
            pred_list.append(rst_data)
    return label_list, pred_list

def compute_auc(labels_list, preds_list):
    """
    The AUC calculation function
    Input:
            labels_list: list of true label
            preds_list:  list of predicted label
    Outputs
            Float, means of AUC
    """
    print("-"*25, "now in function compute_auc", "-"*25)
    auc = []
    if labels_list.shape[0] <= 0:
        return "label list is None!"
    print("shape of labels_list", labels_list.shape)
    print("shape of preds_list", preds_list.shape)
    n_bins = labels_list.shape[0] // 2
    if labels_list.ndim == 1:
        labels_list = labels_list.reshape(-1, 1)
        preds_list = preds_list.reshape(-1, 1)
    for i in range(labels_list.shape[1]):
        labels = labels_list[:, i]
        preds = preds_list[:, i]
        postive_len = labels.sum()
        negative_len = labels.shape[0] - postive_len
        total_case = postive_len * negative_len
        positive_histogram = np.zeros((n_bins))
        negative_histogram = np.zeros((n_bins))
        bin_width = 1.0 / n_bins
        for j, _ in enumerate(labels):
            nth_bin = int(preds[j] // bin_width)
            if nth_bin == n_bins:
                nth_bin = nth_bin - 1
            if labels[j]:
                positive_histogram[nth_bin] = positive_histogram[nth_bin] + 1
            else:
                negative_histogram[nth_bin] = negative_histogram[nth_bin] + 1
        accumulated_negative = 0
        satisfied_pair = 0
        for k in range(n_bins):
            satisfied_pair += (
                positive_histogram[k] * accumulated_negative +
                positive_histogram[k] * negative_histogram[k] * 0.5)
            accumulated_negative += negative_histogram[k]
        auc.append(satisfied_pair / total_case)

    return np.mean(auc)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("ERROR-three arguments are required, your command should be like this:")
        print(" python sdk_get_auc.py  info_file_path info_filename  infer_results_path")
        print("For example:")
        print(" python sdk_get_auc.py  ../data/config/ annotations_final.csv  ../sdk/results/")
    else:
        base_info_path = sys.argv[1]
        info_file_name = sys.argv[2]
        infer_result_path = sys.argv[3]
        simp_info_tags = simplify_tagging_info(base_info_path, info_file_name)
        _label_list, _pred_list = get_labels(simp_info_tags, infer_result_path)
        auc_val = compute_auc(np.array(_label_list), np.array(_pred_list))
        print("-" * 27 + " Validation Performance " + "-" * 27)
        print("AUC: {:.5f}\n".format(auc_val))
