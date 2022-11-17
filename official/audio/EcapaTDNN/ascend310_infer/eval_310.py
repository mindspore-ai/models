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

import os
import sys
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from scipy.spatial.distance import cosine
from mindspore import Tensor
from src.metrics import get_EER_from_scores

def evaluate(spk2emb, utt2emb, trials):
    # Evaluate EER given utterance to embedding mapping and trials file
    scores, labels = [], []
    with open(trials, "r") as f:
        for trial in f:
            trial = trial.strip()
            label, spk, test = trial.split(" ")
            spk = spk[:-4].replace('/', '_')
            if label == '1':
                labels.append(1)
            else:
                labels.append(0)
            enroll_emb = spk2emb[spk]
            test = test[:-4].replace('/', '_')
            test_emb = utt2emb[test]
            scores.append(1 - cosine(enroll_emb, test_emb))

    return get_EER_from_scores(scores, labels)[0]

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        hparams_file = sys.argv[1]
    else:
        hparams_file = "../ecapa-tdnn_config.yaml"
    print("hparam:", hparams_file)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
        print(hparams)
    enroll_dir = "output/"
    enroll_dict = dict()
    with open(os.path.join(enroll_dir, 'emb.txt'), 'r') as fp:
        for line in fp:
            emb_file = enroll_dir + line.strip()
            arr = np.fromfile(emb_file, dtype=np.float32)
            enroll_dict[line[:-5]] = arr

    eer = evaluate(enroll_dict, enroll_dict, os.path.join(
        '../', hparams['veri_file_path']))
    print("eer baseline:", eer)

    glob_mean = Tensor([0])
    cnt = 0
    _, glob_mean, cnt = emb_mean(glob_mean, cnt, enroll_dict)
    _, glob_mean, cnt = emb_mean(glob_mean, cnt, enroll_dict)
    enroll_dict_mean, glob_mean, cnt = emb_mean(glob_mean, cnt, enroll_dict)

    eer = evaluate(enroll_dict_mean, enroll_dict_mean,
                   os.path.join('../', hparams['veri_file_path']))
    print("eer sub mean:", eer)
