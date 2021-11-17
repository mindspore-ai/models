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
Dataset reader for preprocessing and converting dataset into bin.
'''

import io
import os
import argparse
import json

from collections import namedtuple
import six
import numpy as np
from tokenizer import FullTokenizer

def csv_reader(fd, delimiter='\t'):
    """
    load csv file
    """
    def gen():
        for i in fd:
            slots = i.rstrip('\n').split(delimiter)
            if len(slots) == 1:
                yield (slots,)
            else:
                yield slots
    return gen()


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            text = text
        elif isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            text = text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            text = text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
    return text


class BaseReader:
    """BaseReader for classify and sequence labeling task"""

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens

        np.random.seed(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def _read_tsv(self, input_file):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
            headers = next(reader)
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`.
        The convention in BERT/ERNIE is:
         (a) For sequence pairs:
          tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
          type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
         (b) For single sequences:
          tokens:   [CLS] the dog is hairy . [SEP]
          type_ids: 0     0   0   0  0     0 0

        Where "type_ids" are used to indicate whether this is the first
        sequence or the second sequence. The embedding vectors for `type=0` and
        `type=1` were learned during pre-training and are added to the wordpiece
        embedding vector (and position vector). This is not *strictly* necessary
        since the [SEP] token unambiguously separates the sequences, but it makes
        it easier for the model to learn the concept of sequences.

        For classification tasks, the first vector (corresponding to [CLS]) is
        used as as the "sentence vector". Note that this only makes sense because
        the entire model is fine-tuned.
        """
        text_a = convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if "text_b" in example._fields:
            text_b = convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        token_type_id = []
        tokens.append("[CLS]")
        token_type_id.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_id.append(0)
        tokens.append("[SEP]")
        token_type_id.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_id.append(1)
            tokens.append("[SEP]")
            token_type_id.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            token_type_id.append(0)

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['input_ids', 'input_mask', 'token_type_id', 'label_id'])

        record = Record(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_id=token_type_id,
            label_id=label_id)
        return record

    def get_num_examples(self, input_file):
        """return total number of examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def get_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return examples


class ClassifyReader(BaseReader):
    """ClassifyReader"""

    def _read_tsv(self, input_file):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text.replace(' ', '')
                example = Example(*line)
                examples.append(example)
            return examples

    def get_all_path(self, output_path):
        """
        Args:
            output_path: save path of convert dataset
        Returns:
            the path of ids, mask, token, label
        """
        ids_path = os.path.join(output_path, "00_data")
        mask_path = os.path.join(output_path, "01_data")
        token_path = os.path.join(output_path, "02_data")
        label_path = os.path.join(output_path, "03_data")
        for path in [ids_path, mask_path, token_path, label_path]:
            os.makedirs(path, 0o755, exist_ok=True)

        return ids_path, mask_path, token_path, label_path

    def file_based_convert_examples_to_features(self, input_file, output_file):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._read_tsv(input_file)
        output_ids, output_mask, output_token, output_label = self.get_all_path(output_file)
        example_count = 0
        for _, example in enumerate(examples):
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer)
            file_name = args_opt.task_type + "_" + str(example_count) + ".bin"
            ids_file_path = os.path.join(output_ids, file_name)
            np.array(record.input_ids, dtype=np.int32).tofile(ids_file_path)

            mask_file_path = os.path.join(output_mask, file_name)
            np.array(record.input_mask, dtype=np.int32).tofile(mask_file_path)

            token_file_path = os.path.join(output_token, file_name)
            np.array(record.token_type_id, dtype=np.int32).tofile(token_file_path)

            label_file_path = os.path.join(output_label, file_name)
            np.array(record.label_id, dtype=np.int32).tofile(label_file_path)
            example_count += 1
            if example_count % 3000 == 0:
                print(example_count)
        print("total example:", example_count)


class SequenceLabelingReader(BaseReader):
    """SequenceLabelingReader"""

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""
        tokens = convert_to_unicode(example.text_a).split(u"")
        labels = convert_to_unicode(example.label).split(u"")
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id] + [self.label_map[label] for label in labels] + [no_entity_id]
        input_mask = [1] * len(input_ids)
        token_type_id = [0] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            token_type_id.append(0)
            label_ids.append(no_entity_id)

        Record = namedtuple(
            'Record',
            ['input_ids', 'input_mask', 'token_type_id', 'label_ids'])

        record = Record(
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_id=token_type_id,
            label_ids=label_ids)
        return record

    def _reseg_token_label(self, tokens, labels, tokenizer):
        """reseg token label."""
        assert len(tokens) == len(tokens)
        ret_tokens, ret_labels = [], []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if not sub_token:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))
            elif label.startswith("S-"):
                b_laebl = "B-" + label[2:]
                e_label = "E-" + label[2:]
                i_label = "I-" + label[2:]
                ret_labels.extend([b_laebl] + [i_label] * (len(sub_token) - 2) + [e_label])
            elif label.startswith("E-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([i_label] * (len(sub_token) - 1) + [label])

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def file_based_convert_examples_to_features(self, input_file, output_file):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        examples = self._read_tsv(input_file)

        writer = FileWriter(file_name=output_file, shard_num=1)
        nlp_schema = {
            "input_ids": {"type": "int64", "shape": [-1]},
            "input_mask": {"type": "int64", "shape": [-1]},
            "token_type_id": {"type": "int64", "shape": [-1]},
            "label_ids": {"type": "int64", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed classification dataset")
        data = []
        for index, example in enumerate(examples):
            if index % 10000 == 0:
                logging.info("Writing example %d of %d" % (index, len(examples)))
            record = self._convert_example_to_record(example, self.max_seq_len, self.tokenizer)
            sample = {
                "input_ids": np.array(record.input_ids, dtype=np.int64),
                "input_mask": np.array(record.input_mask, dtype=np.int64),
                "token_type_id": np.array(record.token_type_id, dtype=np.int64),
                "label_ids": np.array([record.label_ids], dtype=np.int64),
            }
            data.append(sample)
        writer.write_raw_data(data)
        writer.commit()


reader_dict = {
    'chnsenticorp': ClassifyReader,
    'msra_ner': SequenceLabelingReader,
    'xnli': ClassifyReader,
    'dbqa': ClassifyReader
}


have_label_map = {
    'chnsenticorp': False,
    'msra_ner': True,
    'xnli': True,
    'dbqa': False
}


def main():
    reader = reader_dict[args_opt.task_type](
        vocab_path=args_opt.vocab_file,
        label_map_config=args_opt.label_map_config if have_label_map[args_opt.task_type] else None,
        max_seq_len=args_opt.max_seq_len,
        do_lower_case=args_opt.do_lower_case,
        random_seed=args_opt.random_seed
    )
    reader.file_based_convert_examples_to_features(input_file=args_opt.data_path, output_file=args_opt.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read dataset and save it to bin")
    parser.add_argument("--task_type", type=str, default="", help="task type to preprocess")
    parser.add_argument("--vocab_file", type=str, default="", help="vocab file")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--data_path", type=str, default="valid.json", help="raw data file")
    parser.add_argument("--output_path", type=str, default="./data", help="the path of convert dataset.")
    parser.add_argument("--label_map_config", type=str, default=None, help="label mapping config file")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed number")

    args_opt = parser.parse_args()
    main()
