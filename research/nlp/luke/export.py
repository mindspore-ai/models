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
"""export checkpoint file into air models"""

import numpy as np

from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import export

from src.luke.config import LukeConfig
from src.reading_comprehension.model import LukeForReadingComprehension
from src.utils.model_utils import ModelArchive
from src.model_utils.config_args import args_config as args

# load pretrain
model_archive = ModelArchive.load(args.model_file)
args.bert_model_name = model_archive.bert_model_name
args.max_mention_length = model_archive.max_mention_length
args.model_path = model_archive.model_path
luke_config = LukeConfig(**model_archive.metadata["model_config"])
args.model_config = luke_config


def run_export():
    """export fun"""
    network = LukeForReadingComprehension(luke_config)
    param_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(network, param_dict)
    file_name = args.file_name
    word_ids = Tensor(np.ones((args.export_batch_size, args.max_seq_length)).astype(np.int32))
    word_segment_ids = Tensor(np.ones((args.export_batch_size, args.max_seq_length)).astype(np.int32))
    word_attention_mask = Tensor(np.ones((args.export_batch_size, args.max_seq_length)).astype(np.int32))
    entity_ids = Tensor(np.ones((args.export_batch_size, args.max_entity_length)).astype(np.int32))
    entity_position_ids = Tensor(
        np.ones((args.export_batch_size, args.max_entity_length, args.max_mention_length)).astype(np.int32))
    entity_segment_ids = Tensor(np.ones((args.export_batch_size, args.max_entity_length)).astype(np.int32))
    entity_attention_mask = Tensor(np.ones((args.export_batch_size, args.max_entity_length)).astype(np.int32))
    export(network, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
           entity_segment_ids, entity_attention_mask, file_name=file_name, file_format='MINDIR')


if __name__ == '__main__':
    run_export()
