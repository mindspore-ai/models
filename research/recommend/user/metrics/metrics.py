# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

# Source Reference: https://github.com/lileipisces/NLG4RS

import math

from rouge import rouge
from bleu import compute_bleu


###############################################################################
# Recommendation Metric
###############################################################################


def mean_absolute_error(predicted, max_r, min_r, mae=True):  # MSE ↓
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += (sub ** 2)

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):  # RMSE ↓
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


def evaluate_ndcg(user2items_test, user2items_top):
    top_k = len(list(user2items_top.values())[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    for u, test_items in user2items_test.items():
        rank_list = user2items_top[u]
        dcg_u = 0
        for idx, item in enumerate(rank_list):
            if item in test_items:
                dcg_u += dcgs[idx]
        ndcg += dcg_u

    return ndcg / (sum(dcgs) * len(user2items_test))


def evaluate_precision_recall_f1(user2items_test, user2items_top):
    top_k = len(list(user2items_top.values())[0])

    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    for u, test_items in user2items_test.items():
        rank_list = user2items_top[u]
        hits = len(test_items & set(rank_list))
        pre = hits / top_k
        rec = hits / len(test_items)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    precision = precision_sum / len(user2items_test)
    recall = recall_sum / len(user2items_test)
    f1 = f1_sum / len(user2items_test)

    return precision, recall, f1


###############################################################################
# Text Metrics
###############################################################################

def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):  # USR ↑
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):  # FMR ↑
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):  # FCR ↑
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):  # DIV ↓
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator
