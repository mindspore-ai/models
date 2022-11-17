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
'''
metric
'''
import sys
import math
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " eval_file")
    print("eval file format: pred_response \t gold_response")
    exit()

def get_dict(tokens, ngram, gdict=None):
    """
    get dict
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def count(pred_tokens, gold_tokens, ngram, result):
    """
    count
    """
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count


def calc_bp(pair_list):
    """
    calc_bp
    """
    c_count = 0.0
    r_count = 0.0
    for pair in pair_list:
        pred_tokens1, gold_tokens1 = pair
        c_count += len(pred_tokens1)
        r_count += len(gold_tokens1)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp


def calc_cover_rate(pair_list, ngram):
    """
    calc_cover_rate
    """
    result = [0.0, 0.0] # [cover_count, total_count]
    for pair in pair_list:
        pred_tokens2, gold_tokens2 = pair
        count(pred_tokens2, gold_tokens2, ngram, result)
    cover_rate = result[0] / result[1]
    return cover_rate


def calc_bleu(pair_list):
    """
    calc_bleu
    """
    bp = calc_bp(pair_list)
    cover_rate1 = calc_cover_rate(pair_list, 1)
    cover_rate2 = calc_cover_rate(pair_list, 2)
    bleu1 = 0
    bleu2 = 0

    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    return [bleu1, bleu2]


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for _, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
        #if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1_ = calc_distinct_ngram(pair_list, 1)
    distinct2_ = calc_distinct_ngram(pair_list, 2)
    return [distinct1_, distinct2_]


def calc_f1(data):
    """
    calc_f1
    """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in data:
        golden_response = "".join(golden_response)
        response = "".join(response)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    f1_ = 2 * p * r / (p + r)
    return f1_


eval_file = sys.argv[1]
sents = []
for line in open(eval_file):
    tk = line.strip().split("\t")
    if len(tk) < 2:
        continue
    pred_tokens_ = tk[0].strip().split(" ")
    gold_tokens_ = tk[1].strip().split(" ")
    sents.append([pred_tokens_, gold_tokens_])
# calc f1
f1 = calc_f1(sents)
# calc bleu
bleu1_, bleu2_ = calc_bleu(sents)
# calc distinct
distinct1, distinct2 = calc_distinct(sents)

output_str = "F1: %.2f%%\n" % (f1 * 100)
output_str += "BLEU1: %.3f%%\n" % bleu1_
output_str += "BLEU2: %.3f%%\n" % bleu2_
output_str += "DISTINCT1: %.3f%%\n" % distinct1
output_str += "DISTINCT2: %.3f%%\n" % distinct2
sys.stdout.write(output_str)
