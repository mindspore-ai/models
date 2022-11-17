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
"""
File: extract_predict_utterance.py
"""

import sys
import json
import collections


def extract_predict_utterance(sample_file, score_file, output_file):
    """
    convert_result_for_eval
    """
    sample_list = [line.strip() for line in open(sample_file, 'r')]
    score_list = [line.strip() for line in open(score_file, 'r')]

    fout = open(output_file, 'w')
    index = 0
    for _, sample in enumerate(sample_list):
        sample = json.loads(sample, encoding="utf-8", \
                              object_pairs_hook=collections.OrderedDict)

        candidates = sample["candidate"]
        scores = score_list[index: index + len(candidates)]

        pridict = candidates[0]
        max_score = float(scores[0])
        for j, score in enumerate(scores):
            score = float(score)
            if score > max_score:
                pridict = candidates[j]
                max_score = score

        if "response" in sample:
            response = sample["response"]
            fout.write(pridict + "\t" + response + "\n")
        else:
            fout.write(pridict + "\n")

        index = index + len(candidates)

    fout.close()


def main():
    """
    main
    """
    extract_predict_utterance(sys.argv[1],
                              sys.argv[2],
                              sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program earlier!")
