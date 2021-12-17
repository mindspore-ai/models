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
""" postprocess. """
import os
import numpy as np

from src.model_utils.config import config


def eval_postprocess(positive_path, negative_type):
    """ get accuracy """
    files = os.listdir(positive_path)
    log = []
    for f in files:
        score_file = os.path.join(config.result_path, f.split('.')[0] + '_0.bin')
        positive_file = os.path.join(positive_path, f)
        argsort = np.fromfile(score_file, np.int32)

        if negative_type == 'head':
            positive_arg = np.fromfile(positive_file, np.int32)[0]  # 0 is head
        else:
            positive_arg = np.fromfile(positive_file, np.int32)[2]  # 2 is tail

        ranking = np.where(argsort == positive_arg)[0][0]
        ranking = 1 + ranking
        log.append({
            'MRR': 1.0 / ranking,
            'MR': ranking,
            'HITS@1': 1.0 if ranking <= 1 else 0.0,
            'HITS@3': 1.0 if ranking <= 3 else 0.0,
            'HITS@10': 1.0 if ranking <= 10 else 0.0,
        })
    return log


def run_eval():
    logs = []

    logs += eval_postprocess(config.head_positive_path, 'head')
    logs += eval_postprocess(config.tail_positive_path, 'tail')

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    print(metrics)


if __name__ == '__main__':
    run_eval()
