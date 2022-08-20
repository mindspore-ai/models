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
Downstream tasks
"""
import os
import json
from collections import defaultdict

import pandas as pd


def read_jsonl(data_path):
    """
    Load the json lines from the specific file.
    Args:
        data_path: The json file path.

    Returns:
        The read json data.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            lines.append(json.loads(line.strip()))
        return lines


def load_qa_dataset_c3(data_dir, split, tokenizer):
    if split.lower() in ('train',):
        train_path = os.path.join(data_dir, 'c3-d-train.json')
        source_data = pd.read_csv(open(train_path, 'r'))
    elif split.lower() in ('validation',):
        dev_path = os.path.join(data_dir, 'c3-d-dev.json')
        source_data = json.load(open(dev_path, 'r'))
    elif split.lower() in ('test',):
        source_data = []
    else:
        raise ValueError(f"The split {split} is not supported. "
                         f"Current only supports 'train', 'dev', 'validation'")

    examples = []
    total_count = 0
    for instance in source_data:
        context = "".join(instance[0])
        queries = instance[1]
        for query in queries:
            question = query['question']
            choices = query['choice']
            answer_true = query['answer']
            total_count += 1
            for choice in choices:
                query_text = f"问：{question}\n答：{choice}\n该答案来自对话：{context}"
                input_str = f"{query_text}"
                input_str.replace('?', '？')
                prompt = f"问：{question}\n答：{choice}\n该答案来自对话："
                prompt.replace('?', '？')
                examples.append({
                    "idx": total_count,
                    "input_str": input_str,
                    "prompt": prompt,
                    "is_correct": answer_true == choice,
                })
    return examples


def load_dataset(dataset, data_url, split='validation', tokenizer=None):
    examples = []
    if dataset == 'c3':
        examples = load_qa_dataset_c3(data_url, split, tokenizer=tokenizer)
    else:
        raise ValueError(f"The eval task {dataset} is not supported now. Currently only support c3.")

    return examples


def get_c3_metric(examples):
    metric = {"top1_acc": 0}
    acc_top1 = 0
    total_count = 0
    score_on_each_example = defaultdict(list)
    for item in examples:
        predicted = item['predict']  # should be score
        idx = item['idx']
        score_on_each_example[idx].append((predicted, item['is_correct']))

    for k in score_on_each_example.keys():
        made_choices = score_on_each_example[k]
        predicted_choice = min(made_choices, key=lambda x: x[0])
        if predicted_choice[1]:
            acc_top1 += 1
        total_count += 1
    metric['top1_acc'] = acc_top1 / total_count
    return metric


def load_metric(dataset):
    if dataset in ('c3',):
        return get_c3_metric
    raise ValueError(f"The input dataset {dataset} not found in the list ['c3']")
