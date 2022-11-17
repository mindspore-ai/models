# Copyright 2021 Huawei Technologies Co., Ltd
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
'''
Data transfer
'''
import argparse
import numpy as np
from mindspore.mindrecord import FileWriter
from mindspore.log import logging

def load_dict(vocab_path):
    """
    load vocabulary dict
    """
    vocab_dict = {}
    idx = 0
    for line in open(vocab_path):
        line = line.strip()
        vocab_dict[line] = idx
        idx += 1
    return vocab_dict

class DataProcessor:
    '''
    transfer data to mindrecord
    '''
    def __init__(self, task_name, vocab_path, max_seq_len, do_lower_case):
        self.task_name = task_name
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case
        self.vocab_dict = load_dict(vocab_path)

    def get_labels(self):
        return ["0", "1"]

    def _read_data(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n').split('\t')
                lines.append(line)
        return lines

    def _create_examples(self, input_file):
        """Creates examples for the training and dev sets."""
        examples = []
        lines = self._read_data(input_file)
        for line in lines:
            context_text = line[1]
            label_text = line[0]
            response_text = line[2]
            if 'kn' in self.task_name:
                kn_text = "%s [SEP] %s" % (line[3], line[4])
            else:
                kn_text = None
            examples.append(
                InputExample(context_text=context_text, response_text=response_text, \
                             kn_text=kn_text, label_text=label_text))
        return examples

    def _convert_example_to_record(self, example, labels, max_seq_len, vocab_dict):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        feature = convert_single_example(example, labels, max_seq_len, vocab_dict)
        return feature

    def file_based_convert_examples_to_features(self, input_file, output_file):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._create_examples(input_file)

        writer = FileWriter(file_name=output_file, shard_num=1)
        nlp_schema = {
            "context_id": {"type": "int64", "shape": [-1]},
            "context_segment_id": {"type": "int64", "shape": [-1]},
            "context_pos_id": {"type": "int64", "shape": [-1]},
            "labels_list": {"type": "int64", "shape": [-1]}
        }
        if 'kn' in self.task_name:
            nlp_schema['kn_id'] = {"type": "int64", "shape": [-1]}
            nlp_schema['kn_seq_length'] = {"type": "int64", "shape": [-1]}

        writer.add_schema(nlp_schema, "proprocessed dataset")
        data = []
        for index, example in enumerate(examples):
            if index % 10000 == 0:
                logging.info("Writing example %d of %d" % (index, len(examples)))
            record = self._convert_example_to_record(example, self.get_labels(), self.max_seq_len, self.vocab_dict)
            sample = {
                "context_id": np.array(record.context_ids, dtype=np.int64),
                "context_pos_id": np.array(record.context_pos_ids, dtype=np.int64),
                "context_segment_id": np.array(record.segment_ids, dtype=np.int64),
                "labels_list": np.array([record.label_ids], dtype=np.int64),
            }
            if 'kn' in self.task_name:
                sample['kn_id'] = np.array(record.kn_ids, dtype=np.int64)
                sample['kn_seq_length'] = np.array(record.kn_seq_length, dtype=np.int64)

            data.append(sample)
        writer.write_raw_data(data)
        writer.commit()

class InputExample():
    """A single training/test example"""

    def __init__(self, context_text, response_text, kn_text, label_text):
        self.context_text = context_text
        self.response_text = response_text
        self.kn_text = kn_text
        self.label_text = label_text

class InputFeatures():
    """input features data"""
    def __init__(self, context_ids, context_pos_ids, segment_ids, kn_ids, kn_seq_length, label_ids):
        self.context_ids = context_ids
        self.context_pos_ids = context_pos_ids
        self.segment_ids = segment_ids
        self.kn_ids = kn_ids
        self.kn_seq_length = kn_seq_length
        self.label_ids = label_ids

def convert_tokens_to_ids(tokens, vocab_dict):
    """
    convert input ids
    """
    ids = []
    for token in tokens:
        if token in vocab_dict:
            ids.append(vocab_dict[token])
        else:
            ids.append(vocab_dict['[UNK]'])
    return ids

def convert_single_example(example, label_list, max_seq_length, vocab_dict):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    if example.context_text:
        tokens_context = example.context_text
        tokens_context = tokens_context.split()
    else:
        tokens_context = []

    if example.response_text:
        tokens_response = example.response_text
        tokens_response = tokens_response.split()
    else:
        tokens_response = []

    if example.kn_text:
        tokens_kn = example.kn_text
        tokens_kn = tokens_kn.split()
        tokens_kn = tokens_kn[0: min(len(tokens_kn), max_seq_length)]
    else:
        tokens_kn = []

    tokens_response = tokens_response[0: min(50, len(tokens_response))]
    if len(tokens_context) > max_seq_length - len(tokens_response) - 3:
        tokens_context = tokens_context[len(tokens_context) \
                + len(tokens_response) - max_seq_length + 3:]

    context_tokens = []
    segment_ids = []

    context_tokens.append("[CLS]")
    segment_ids.append(0)
    context_tokens.extend(tokens_context)
    segment_ids.extend([0] * len(tokens_context))
    context_tokens.append("[SEP]")
    segment_ids.append(0)

    context_tokens.extend(tokens_response)
    segment_ids.extend([1] * len(tokens_response))
    context_tokens.append("[SEP]")
    segment_ids.append(1)

    context_ids = convert_tokens_to_ids(context_tokens, vocab_dict)
    context_ids = context_ids + [0] * (max_seq_length - len(context_ids))
    context_pos_ids = list(range(len(context_ids))) + [0] * (max_seq_length - len(context_ids))
    segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))
    label_ids = label_map[example.label_text]
    if tokens_kn:
        kn_ids = convert_tokens_to_ids(tokens_kn, vocab_dict)
        kn_ids = kn_ids[0: min(max_seq_length, len(kn_ids))]
        kn_seq_length = len(kn_ids)
        kn_ids = kn_ids + [0] * (max_seq_length - kn_seq_length)
    else:
        kn_ids = []
        kn_seq_length = 0

    # print(len(context_ids), len(context_pos_ids), len(segment_ids), len(kn_ids), kn_seq_length, label_ids)
    feature = InputFeatures(
        context_ids=context_ids,
        context_pos_ids=context_pos_ids,
        segment_ids=segment_ids,
        kn_ids=kn_ids,
        kn_seq_length=kn_seq_length,
        label_ids=label_ids)

    return feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--task_name", type=str, default="match", choices=["match", "match_kn", "match_kn_gene"],
                        help="vocab file")
    parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--do_lower_case", type=str, default="true",
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")

    parser.add_argument("--input_file", type=str, default="", help="raw data file")
    parser.add_argument("--output_file", type=str, default="", help="minddata file")
    args_opt = parser.parse_args()

    processor = DataProcessor(args_opt.task_name, args_opt.vocab_path, args_opt.max_seq_len,
                              bool(args_opt.do_lower_case == "true"))
    processor.file_based_convert_examples_to_features(args_opt.input_file, args_opt.output_file)
