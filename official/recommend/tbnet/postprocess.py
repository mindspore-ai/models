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
"""preprocess data"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Postprocess of Hypertext Inference')
parser.add_argument('--result_Path', type=str, default='./result_Files',
                    help='result path')
parser.add_argument('--label_Path', default='./result_Files', type=str,
                    help='label file path')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
args = parser.parse_args()

def calculate_auc(labels_list, preds_list):
    """
    The AUC calculation function
    Input:
            labels_list: list of true label
            preds_list:  list of predicted label
    Outputs
            Float, means of AUC
    """
    auc = []
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

dirs = os.listdir(args.label_Path)
cur, total = 0, 0
print('---------- start cal acc ----------')
gt_list = []
pred_list = []
for file in dirs:
    label = np.fromfile(os.path.join(args.label_Path, file), dtype=np.float32)
    gt_list.append(label)

    file_name = file.split('.')[0]
    idx = file_name.split('_')[-1]
    predict_file_name = "tbnet_item_bs1_" + str(idx) + "_1.bin"
    predict_file = os.path.join(args.result_Path, predict_file_name)
    predict = np.fromfile(predict_file, dtype=np.float32)
    pred_list.append(predict)
res_pred = np.concatenate(pred_list, axis=0)
res_true = np.concatenate(gt_list, axis=0)
rst_auc = calculate_auc(res_true, res_pred)
print('auc:', rst_auc)
