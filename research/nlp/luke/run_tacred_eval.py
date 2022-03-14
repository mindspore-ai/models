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
Relation classification eval script
"""

import os

from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net

from src.luke.config import LukeConfig
from src.model_utils.config_args import args_config as args
from src.relation_classification.model import LukeForRelationClassificationEval
from src.relation_classification.preprocess_data import load_eval
from src.utils.model_utils import ModelArchive

context.set_context(mode=context.GRAPH_MODE, device_target=args.device)


# load pretrain

def evaluate(arg, model):
    model.set_train(False)
    dataset, _ = load_eval(arg)  # set of labels
    predictions = []
    labels = []

    for batch in dataset:
        label_ = sum(batch[7].asnumpy().tolist(), [])
        logits = model(*batch[:-1])
        predictions.extend(logits.asnumpy().argmax(axis=1))
        labels.extend(label_)
    res = calculate_metric(labels, predictions)
    return res


def calculate_metric(labels, predictions):
    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0
    for label, prediction in zip(labels, predictions):
        if prediction != 0:
            num_predicted_labels += 1
        if label != 0:
            num_gold_labels += 1
            if prediction == label:
                num_correct_labels += 1

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0
    recall = num_correct_labels / num_gold_labels
    if recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(precision=precision, recall=recall, f1=f1)


def do_eval():
    """do eval"""
    args.data_dir = os.path.join(args.data, 'tacred_change')
    model_archive = ModelArchive.load(args.model_file)
    luke_config = LukeConfig(**model_archive.metadata["model_config"])
    args.model_config = luke_config
    args.model_config.entity_vocab_size = 3
    args.model_config.vocab_size += 2

    network = LukeForRelationClassificationEval(luke_config, num_labels=42)
    if os.path.isdir(args.checkpoint_file):
        ckpt_list = os.listdir(args.checkpoint_file)
        ckpt_list = list(filter(lambda x: x.endswith('.ckpt'), ckpt_list))
        ckpt_list = [os.path.join(args.checkpoint_file, ckpt) for ckpt in ckpt_list]
    else:
        ckpt_list = [args.checkpoint_file]

    for checkpoint in ckpt_list:
        print("checkpoint name: ", checkpoint)
        param_dict = load_checkpoint(checkpoint)
        load_param_into_net(network, param_dict)
        print("evaluating...", flush=True)
        res = evaluate(args, network)
        print(res, flush=True)


if __name__ == '__main__':
    do_eval()
