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
"""Sorted pic values"""
import os
import json
import argparse
from collections import defaultdict
import numpy as np


def cal_entropy(logist):
    """according the logist get the entropy"""
    info_entro = 0.0
    for each_probe in logist:
        if each_probe == 0:
            continue
        info_entro = info_entro - each_probe * np.log2(each_probe)
    return info_entro


def norm_dict(_dict):
    """according the dict max, do norm"""
    value_max = max(_dict.values())
    for key, value in _dict.items():
        _dict[key] = value / value_max
    return _dict


def caculate_value(infer_result, weight=(1., 1., 1.)):
    '''calculate the value of each img'''

    difficult_indicators = defaultdict(float)
    information_indicators = defaultdict(float)
    diversity_indicators = defaultdict(float)

    # calculate each img difficult/information/diversity
    for image_path, boxes_info in infer_result.items():
        _difficult = 0.
        _information = 0.
        _diversity = 0.

        _cls_set = set()
        _difficult_list = []
        for box in boxes_info:
            entropy = cal_entropy(box['pred score'])
            _difficult_list.append(entropy)
            _information += box['confidence score']
            _cls_set.add(box['pred class'])

        _diversity = len(_cls_set)
        try:
            _difficult = sum(_difficult_list) / len(_difficult_list)
        except ZeroDivisionError:
            _difficult = 99

        difficult_indicators[image_path] = _difficult
        information_indicators[image_path] = _information
        diversity_indicators[image_path] = _diversity

    # do norm
    _difficult_indicators = norm_dict(difficult_indicators)
    _information_indicators = norm_dict(information_indicators)
    _diversity_indicators = norm_dict(diversity_indicators)

    tmp_list = []
    for key in _difficult_indicators.keys():
        _final_value = _difficult_indicators[key] * weight[0] + \
                       _information_indicators[key] * weight[1] + \
                       _diversity_indicators[key] * weight[2]
        tmp_list.append([key, _final_value])
    sorted_list = sorted(tmp_list, key=lambda x: x[1], reverse=True)
    return sorted_list


def run_select(cfg):
    with open(cfg.infer_json, 'r') as file:
        infer_results = json.load(file)
    he_list = caculate_value(infer_results)
    he_dict = {he[0]: he[1] for he in he_list}
    with open(os.path.join(cfg.exp_dir, "top_value_data.json"), "w") as file:
        json.dump(he_dict, file)
    return he_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate annotation')
    parser.add_argument('--infer_json', type=str, help='the output of infer_combine')
    parser.add_argument("--exp_dir", default='./', type=str, help="txt save path")

    args = parser.parse_args()
    run_select(args)
