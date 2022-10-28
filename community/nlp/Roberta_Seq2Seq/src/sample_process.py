# Copyright 2020 Huawei Technologies Co., Ltd
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
""" sample process """
from nltk.tokenize import word_tokenize


def process_one_example(tokenizer, text, max_seq_len=128):
    """process one testline"""
    tokens = word_tokenize(text)
    # tokens=tokenizer.tokenize(text)
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
    ntokens = []
    ntokens.append("<s>")
    for _, token in enumerate(tokens):
        ntokens.append(token)
    ntokens.append("</s>")
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    attention_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(1)
        attention_mask.append(0)
        ntokens.append("<pad>")  # pad 1
    assert len(input_ids) == max_seq_len
    assert len(attention_mask) == max_seq_len
    feature = (input_ids, attention_mask)
    return feature
