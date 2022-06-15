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
Export tacred checkpoint file into MINDIR format
"""

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.luke.config import LukeConfig
from src.model_utils.config_args import args_config as args
from src.relation_classification.model import LukeForRelationClassificationEval
from src.utils.model_utils import ModelArchive

context.set_context(mode=context.GRAPH_MODE, device_target=args.device)

# load pretrain
model_archive = ModelArchive.load(args.model_file)
args.bert_model_name = model_archive.bert_model_name
args.max_mention_length = model_archive.max_mention_length
args.model_path = model_archive.model_path
luke_config = LukeConfig(**model_archive.metadata["model_config"])
args.model_config = luke_config
args.model_config.entity_vocab_size = 3
args.model_config.vocab_size += 2


def run_export(file_name, file_format):
    """export fun"""
    network = LukeForRelationClassificationEval(luke_config, num_labels=42)
    param_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(network, param_dict)
    word_ids = Tensor(np.ones((args.export_batch_size, args.max_seq_length)).astype(np.int32))
    word_segment_ids = Tensor(np.ones((args.export_batch_size, args.max_seq_length)).astype(np.int32))
    word_attention_mask = Tensor(np.ones((args.export_batch_size, args.max_seq_length)).astype(np.int32))
    entity_ids = Tensor(np.ones((args.export_batch_size, args.max_entity_length)).astype(np.int32))
    entity_position_ids = Tensor(
        np.ones((args.export_batch_size, args.max_entity_length, args.max_mention_length)).astype(np.int32))
    entity_segment_ids = Tensor(np.ones((args.export_batch_size, args.max_entity_length)).astype(np.int32))
    entity_attention_mask = Tensor(np.ones((args.export_batch_size, args.max_entity_length)).astype(np.int32))
    export(network, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
           entity_segment_ids, entity_attention_mask, file_name=file_name, file_format=file_format)


if __name__ == '__main__':
    run_export(file_name=args.file_name, file_format=args.file_format)
