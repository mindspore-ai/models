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
"""create squad train data"""
import os

from src.luke.config import LukeConfig
from src.model_utils.config_args import args_config as args
from src.utils.model_utils import ModelArchive, get_entity_vocab_file_path
from src.utils.utils import create_dir_not_exist
from src.utils.word_tokenizer import AutoTokenizer
from src.utils.entity_vocab import EntityVocab
from src.reading_comprehension.dataProcessing import build_data_change

args.wiki_link_db_file = os.path.join(args.wikipedia, "enwiki_20160305.pkl")
args.model_redirects_file = os.path.join(args.wikipedia, "enwiki_20181220_redirects.pkl")
args.link_redirects_file = os.path.join(args.wikipedia, "enwiki_20160305_redirects.pkl")

args.data_dir = os.path.join(args.data, 'squad')

create_dir_not_exist(args.data)
create_dir_not_exist(args.data_dir)

model_archive = ModelArchive.load(args.model_file)
args.bert_model_name = model_archive.bert_model_name
args.max_mention_length = model_archive.max_mention_length
args.model_path = model_archive.model_path
luke_config = LukeConfig(**model_archive.metadata["model_config"])
args.model_config = luke_config
entity_vocab = EntityVocab(get_entity_vocab_file_path(args.model_file))
args.entity_vocab = entity_vocab
args.tokenizer = AutoTokenizer.from_pretrained(model_archive.bert_model_name)
build_data_change(args)
