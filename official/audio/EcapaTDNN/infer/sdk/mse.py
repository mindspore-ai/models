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

import argparse
import numpy as np
from scipy.spatial.distance import cosine

def evaluate(spk2emb, utt2emb, trials):
    # Evaluate EER given utterance to embedding mapping and trials file
    scores, labels = [], []
    with open(trials, "r") as f:
        for trial in f:
            trial = trial.strip()
            label, spk, test = trial.split(" ")
            spk = spk[:-4]
            if label == '1':
                labels.append(1)
            else:
                labels.append(0)
            enroll_emb = spk2emb[spk]
            test_emb = utt2emb[test[:-4]]
            scores.append(1 - cosine(enroll_emb, test_emb))

    return get_EER_from_scores(scores, labels)[0]

def get_EER_from_scores(scores, labels, pos_label=1):
    """Compute EER given scores and labels
    """
    P_fa, P_miss, thresholds = compute_fa_miss(scores, labels, pos_label, return_thresholds=True)
    eer, thresh_eer = get_EER(P_fa, P_miss, thresholds)
    return eer, thresh_eer

def compute_fa_miss(scores, labels, pos_label=1, return_thresholds=True):
    """Returns P_fa, P_miss, [thresholds]
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    P_fa = fpr[::-1]
    P_miss = 1. - tpr[::-1]
    thresholds = thresholds[::-1]
    if return_thresholds:
        return P_fa, P_miss, thresholds
    return P_fa, P_miss

def get_EER(P_fa, P_miss, thresholds=None):
    """Compute EER given false alarm and miss probabilities
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x: x - interp1d(P_fa, P_miss)(x), 0., 1.)
    eer = float(eer)
    if thresholds is None:
        return eer
    thresh_eer = interp1d(P_fa, thresholds)(eer)
    thresh_eer = float(thresh_eer)
    return eer, thresh_eer

def emb_mean(g_mean, increment, emb_dict):
    emb_dict_mean = dict()
    for utt in emb_dict:
        if increment == 0:
            g_mean = emb_dict[utt]
        else:
            weight = 1 / (increment + 1)
            g_mean = (
                1 - weight
            ) * g_mean + weight * emb_dict[utt]
        emb_dict_mean[utt] = emb_dict[utt] - g_mean
        increment += 1
        if increment % 3000 == 0:
            print('processing ', increment)
    return emb_dict_mean, g_mean, increment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_path', type=str, default='../output/enroll_dict_bleeched.npy')
    parser.add_argument('--veri_file_path', type=str, default='../feat_eval/veri_test_bleeched.txt')
    hparams = parser.parse_args()
    npy_path = hparams.npy_path
    veri_file_path = hparams.veri_file_path
    enroll_dict = np.load(npy_path, allow_pickle=True)
    eer1 = evaluate(enroll_dict, enroll_dict, veri_file_path)
    print("eer baseline:", eer1)
    print("Sub mean...")
    glob_mean = np.zeros(8)
    cnt = 0
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, enroll_dict)
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, enroll_dict)
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, enroll_dict)
    eer2 = evaluate(enroll_dict_mean, enroll_dict_mean, veri_file_path)
    print("eer with sub mean:", eer2)
