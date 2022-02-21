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
"""
eval skipgram according to model file:
python eval.py --checkpoint_path=[CHECKPOINT_PATH] --dictionary=[ID2WORD_DICTIONARY] &> eval.log &
"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate SkipGram')
parser.add_argument('--eval_data_dir', type=str, default=None, help='evaluation file\'s direcionary.')
args = parser.parse_args()

def cal_top_k_similar(target_embs_, emb_matrix_, ks=1):
    """Return ids of the most similar word of embedding in target_embs
    """
    cosine_projection = np.dot(target_embs_, emb_matrix_.T)
    target_norms = np.linalg.norm(target_embs_, axis=1).reshape(-1, 1)
    emb_norms = np.linalg.norm(emb_matrix_, axis=1).reshape(1, -1)
    cosine_similarity = cosine_projection / (np.dot(target_norms, emb_norms))
    top_k_similar = np.argsort(-cosine_similarity, axis=1)
    top_k_similar = top_k_similar[:, :ks]
    return top_k_similar

def load_eval_data(data_dir):
    """load questions-words.txt
    """
    samples = dict()
    files = os.listdir(data_dir)
    for filename in files:
        data_path = os.path.join(data_dir, filename)
        if not os.path.isfile(data_path):
            continue
        with open(data_path, 'r') as f:
            na = "capital-common-countries"
            samples[na] = list()
            for line in f:
                if ':' in line:
                    strs = line.strip().split(' ')
                    na = strs[1]
                    samples[na] = list()
                else:
                    samples[na].append(line.strip().lower().split(' '))
    return samples

if __name__ == '__main__':

    args.eval_data_dir = "./eval_data"
    w2v_emb_save_dir = "../../temp/w2v_emb"

    print("start load w2v_emb.npy")
    w2v_emb = np.load(os.path.join(w2v_emb_save_dir, 'w2v_emb.npy'), allow_pickle=True).item()
    if args.eval_data_dir is not None:
        samples_ = load_eval_data(args.eval_data_dir)
    else:
        print("eval_data is None")
    emb_list = list(w2v_emb.items())
    emb_matrix = np.array([item[1] for item in emb_list])
    target_embs = []
    labels = []
    ignores = []
    for sample_type in samples_:
        type_k = samples_[sample_type]
        for sample in type_k:
            try:
                vecs = [w2v_emb[w] for w in sample]
            except KeyError:
                continue
            vecs = [vec / np.linalg.norm(vec) for vec in vecs]
            target_embs.append((vecs[1] + vecs[2] - vecs[0]) / 3)
            labels.append(sample[3])
            ignores.append([sample[0], sample[1], sample[2]])
    top_k_similar_ = cal_top_k_similar(np.array(target_embs), emb_matrix, ks=5)

    correct_cnt = 0
    for i, candidate_index in enumerate(top_k_similar_):
        ignore = ignores[i]
        label = labels[i]
        for ci in candidate_index:
            predicted = emb_list[ci][0]
            if predicted not in ignore:
                break
        if predicted == label:
            correct_cnt += 1
        print('predicted: %-15s label: %s'% (predicted, label))
    print("Total Accuracy: %.2f%%"% (correct_cnt / len(target_embs) * 100))
