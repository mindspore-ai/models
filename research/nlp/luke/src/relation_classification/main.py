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
Data process file
"""

import json

from tqdm import tqdm
from transformers import RobertaTokenizer

from src.relation_classification.utils import DatasetProcessor
from src.relation_classification.utils import InputFeatures

HEAD_TOKEN = "[HEAD]"
TAIL_TOKEN = "[TAIL]"


def convert_examples_to_features(examples, label_list, tokenizer, max_mention_length):
    label_map = {l: i for i, l in enumerate(label_list)}

    def tokenize(text):
        text = text.rstrip()
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=True)
        return tokenizer.tokenize(text)

    features = []
    for example in tqdm(examples):
        if example.span_a[1] < example.span_b[1]:
            span_order = ("span_a", "span_b")
        else:
            span_order = ("span_b", "span_a")

        tokens = [tokenizer.cls_token]
        cur = 0
        token_spans = {}
        for span_name in span_order:
            span = getattr(example, span_name)
            tokens += tokenize(example.text[cur: span[0]])
            start = len(tokens)
            tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
            tokens += tokenize(example.text[span[0]: span[1]])
            tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
            token_spans[span_name] = (start, len(tokens))
            cur = span[1]

        tokens += tokenize(example.text[cur:])
        tokens.append(tokenizer.sep_token)

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)

        entity_ids = [1, 2]
        entity_position_ids = []
        for span_name in ("span_a", "span_b"):
            span = token_spans[span_name]
            position_ids = list(range(span[0], span[1]))[:max_mention_length]
            position_ids += [-1] * (max_mention_length - span[1] + span[0])
            entity_position_ids.append(position_ids)

        entity_segment_ids = [0, 0]
        entity_attention_mask = [1, 1]

        features.append(
            InputFeatures(
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                label=label_map[example.label],
            )
        )

    return features


def load_examples(args, fold="train"):
    processor = DatasetProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    print("Creating features from the tacred dataset...")
    features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length)
    a = []
    for feature in features:
        data = {}
        data['word_ids'] = feature.word_ids
        data['word_segment_ids'] = feature.word_segment_ids
        data['word_attention_mask'] = feature.word_attention_mask
        data['entity_ids'] = feature.entity_ids
        data['entity_position_ids'] = feature.entity_position_ids
        data['entity_segment_ids'] = feature.entity_segment_ids
        data['entity_attention_mask'] = feature.entity_attention_mask
        data['label'] = feature.label
        data['entity_ids'] = feature.entity_ids
        a.append(data)
    b = json.dumps(a)
    f2 = open('new_json.json', 'w')
    f2.write(b)
    f2.close()
    print("over")

    def collate_fn(o):
        def create_padded_sequence(o, attr_name, padding_value, max_len):
            """create padding"""
            value = getattr(o[1], attr_name)
            if attr_name == 'entity_position_ids':
                if len(value) > max_len:
                    return value[:max_len]
                res = value + [[padding_value] * len(value[0])] * (max_len - len(value))
                return res
            if len(value) > max_len:
                return value[:max_len]
            return value + [padding_value] * (max_len - len(value))

        return dict(
            word_ids=create_padded_sequence(o, "word_ids", args.tokenizer.pad_token_id, args.max_seq_length),
            word_attention_mask=create_padded_sequence(o, "word_attention_mask", 0, args.max_seq_length),
            word_segment_ids=create_padded_sequence(o, "word_segment_ids", 0, args.max_seq_length),
            entity_ids=create_padded_sequence(o, "entity_ids", 0, args.max_entity_length),
            entity_attention_mask=create_padded_sequence(o, "entity_attention_mask", 0, args.max_entity_length),
            entity_position_ids=create_padded_sequence(o, "entity_position_ids", -1, args.max_entity_length),
            entity_segment_ids=create_padded_sequence(o, "entity_segment_ids", 0, args.max_entity_length),
            label=[getattr(o[1], 'label')],
        )

    dataset = []
    for d in tqdm(list(enumerate(features))):
        dataset.append(collate_fn(d))
    if fold == 'eval':
        return dataset, examples, features, label_list
    return dataset
