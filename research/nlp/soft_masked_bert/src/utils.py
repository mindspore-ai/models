# Copyright 2022 Huawei Technologies Co., Ltd
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
"""utils"""

import os
import json
import sys

def compute_correction_prf(results2, all_predict_true_index, all_gold_index):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if all_predict_true_index[i]:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results2[i][2][j])
                if results2[i][1][j] == results2[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results2[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1
    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if correction_precision + correction_recall == 0:
        correction_f1 = 0
    else:
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall)
    print("The correction result is precision={}, recall={} and F1={}".format(correction_precision, \
    correction_recall, correction_f1))
    return correction_f1

def compute_detection_prf(results1):
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results1:
        src, tgt, predict = item
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, \
    detection_recall, detection_f1))
    return all_predict_true_index, all_gold_index, detection_f1

def compute_corrector_prf(results):
    all_predict_true_index_, all_gold_index_, detection_f1_ = compute_detection_prf(results)
    correction_f1_ = compute_correction_prf(results, all_predict_true_index_, all_gold_index_)
    return detection_f1_, correction_f1_

def compute_sentence_level_prf(results):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = len(results)

    for item in results:
        src, tgt, predict = item
        if src == tgt:
            if tgt == predict:
                TN += 1
            else:
                FP += 1
        else:
            if tgt == predict:
                TP += 1
            else:
                FN += 1

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f'Sentence Level: acc:{acc:.6f}, precision:{precision:.6f}, recall:{recall:.6f}, f1:{f1:.6f}')
    return acc, precision, recall, f1

def get_main_dir():
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    return os.path.join(os.path.dirname(__file__), '..', '..')

def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(get_main_dir(), fn))

def load_json(fp):
    if not os.path.exists(fp):
        return dict()
    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)
