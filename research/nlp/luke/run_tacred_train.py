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
Relation classification train script
"""

import os

from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode

from src.luke.config import LukeConfig
from src.model_utils.config_args import args_config as args
from src.relation_classification.main import HEAD_TOKEN
from src.relation_classification.model import LukeForRelationClassification
from src.relation_classification.preprocess_data import load_train
from src.relation_classification.train import do_train
from src.relation_classification.utils import TAIL_TOKEN
from src.utils.model_utils import ModelArchive
from src.utils.word_tokenizer import AutoTokenizer

context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
device_num = 1
set_seed(args.seed)
if args.distribute:
    init()
    device_num = get_group_size()
    context.set_auto_parallel_context(device_num=device_num,
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)

config = args


def runtrain():
    """run squad train"""
    # load pretrain
    args.data_dir = os.path.join(args.data, 'tacred_change')
    model_archive = ModelArchive.load(args.model_file)
    args.bert_model_name = model_archive.bert_model_name
    args.max_mention_length = model_archive.max_mention_length
    args.model_path = model_archive.model_path
    args.tokenizer = AutoTokenizer.from_pretrained(model_archive.bert_model_name)
    luke_config = LukeConfig(**model_archive.metadata["model_config"])
    args.model_config = luke_config
    args.tokenizer.add_special_tokens(dict(additional_special_tokens=[HEAD_TOKEN, TAIL_TOKEN]))
    args.model_config.entity_vocab_size = 3
    args.model_config.vocab_size += 2
    dataset = load_train(args)
    network = LukeForRelationClassification(luke_config, 42)

    do_train(dataset, network, args)


if __name__ == '__main__':
    runtrain()
