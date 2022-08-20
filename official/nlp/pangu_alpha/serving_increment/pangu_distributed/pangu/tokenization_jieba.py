# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""
Tokenization classes for OpenAI GPT.
source code:https://github.com/huggingface/transformers/blob/main/src/transformers/models/openai/tokenization_openai.py
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sentencepiece as spm
import jieba

class JIEBATokenizer():
    r"""
    Jieba Tokenizer
    """
    def __init__(self, model_file, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = {}
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

        for i in range(self.sp.get_piece_size()):
            self.encoder[self.sp.id_to_piece(i)] = i
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        self.eod_id = self.encoder['<eod>']
        self.eot_id = self.encoder['<eot>']
        self.pad_id = self.encoder['<pad>']

    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        return seg_list

    def convert_tokens_to_ids(self, tokens):
        new_seg = " ".join(tokens)
        return self.sp.encode(new_seg)

    def convert_ids_to_tokens(self, ids):
        return self.sp.id_to_piece(ids)

    def process_tokens(self, text):
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        return self.process_tokens(text)
