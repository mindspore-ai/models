# Copyright 2020 Huawei Technologies Co., Ltd
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
"""rouge score calculate"""

from rouge_score import rouge_scorer


def get_rouge_score(predictions, references, rouge_types=None, use_stemmer=False):
    """calculate the rouge score """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    scores = []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        scores.append(score)

    result = {}
    for key in scores[0]:
        result[key] = list(score[key] for score in scores)

    return result
