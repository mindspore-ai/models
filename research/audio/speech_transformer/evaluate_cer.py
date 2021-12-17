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
""" evaluate CER for model results"""

import json
from pathlib import Path

import jiwer

from src.model_utils.config import config


def evaluate_cer():
    """evaluate CER"""
    remove_non_words = jiwer.RemoveKaldiNonWords()
    remove_space = jiwer.RemoveWhiteSpace()
    preprocessing = jiwer.Compose([remove_non_words, remove_space])

    with Path(config.output_file).open('r') as file:
        output_data = json.load(file)

    total_cer = 0
    for sample in output_data.values():
        res_text = preprocessing(sample['output'])
        res_text = ' '.join(res_text)

        gt_text = preprocessing(sample['gt'])
        gt_text = ' '.join(gt_text)

        cer = jiwer.wer(gt_text, res_text)
        total_cer += cer

    print('Resulting cer is ', (total_cer / len(output_data.values())) * 100)


if __name__ == '__main__':
    evaluate_cer()
