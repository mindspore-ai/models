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
"""Evaluation api."""
import argparse
import numpy as np

from src.utils.eval_score import get_score
from src.utils.dictionary import Dictionary

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="mass process")
    parser.add_argument("--source_ids", type=str, default="../data/input/source_ids.txt", help="source ids path")
    parser.add_argument("--target_ids", type=str, default="../data/input/target_ids.txt", help="target ids path")
    parser.add_argument("--infer_result", type=str, default="../sdk/result/result.txt", help="infer result path")
    parser.add_argument("--vocab_file", type=str, default="../data/config/all_en.dict.bin", help="vocab file path")
    parser.add_argument("--metric", type=str, default="rouge", help="vocab file path")
    args_opt = parser.parse_args()
    return args_opt



if __name__ == '__main__':
    args = parse_args()

    vocab = Dictionary.load_from_persisted_dict(args.vocab_file)

    result = []
    source_sentences = np.loadtxt(args.source_ids, dtype=np.int32).reshape(-1, 64)
    target_sentences = np.loadtxt(args.target_ids, dtype=np.int32).reshape(-1, 64)
    predictions = np.loadtxt(args.infer_result, dtype=np.int32).reshape(-1, 65)
    probs = np.loadtxt(args.infer_result, dtype=np.int32).reshape(-1, 65)

    for inputs, ref, batch_out, batch_probs in zip(source_sentences,
                                                   target_sentences,
                                                   predictions,
                                                   probs):
        if batch_out.ndim == 3:
            batch_out = batch_out[:, 0]

        example = {
            "source": inputs.tolist(),
            "target": ref.tolist(),
            "prediction": batch_out.tolist(),
            "prediction_prob": batch_probs.tolist()
        }
        result.append(example)

    get_score(result, vocab, args.metric)
