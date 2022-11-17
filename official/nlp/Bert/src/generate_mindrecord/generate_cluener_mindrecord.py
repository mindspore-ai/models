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
"""Generate MindRecord for BERT finetuning runner: CLUENER."""
import os
import csv
import json
from argparse import ArgumentParser
import numpy as np
import tokenization
from mindspore.mindrecord import FileWriter

def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert task: CLUENER")
    parser.add_argument("--data_dir", type=str, default="",
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", type=str, default="ner", help="The name of the task to train.")
    parser.add_argument("--vocab_file", type=str, default="",
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="The output directory where the mindrecord will be written.")
    parser.add_argument("--do_lower_case", type=bool, default=True, help="Whether to lower case the input text. "
                                                                         "Should be True for uncased models"
                                                                         "and False for cased models.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", type=bool, default=True,
                        help="Whether to run the model in inference mode on the test set.")
    args_opt = parser.parse_args()
    return args_opt


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample():
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures():
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            lines = []
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
            return lines


class NerProcessor(DataProcessor):
    """Processor for the CLUENER data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")))

    def get_labels(self):
        clue_labels = ['address', 'book', 'company', 'game', 'government', 'movie',
                       'name', 'organization', 'position', 'scene']
        with open(os.path.join(args.output_dir, 'label.txt'), 'w') as rf:
            for label in clue_labels:
                rf.write(label + "\n")
        return ['O'] + [p + '-' + l for p in ['B', 'M', 'E', 'S'] for l in clue_labels]

    def generate_label(self, line, label):
        """generate label"""
        for l, words in line['label'].items():
            for _, indices in words.items():
                for index in indices:
                    if index[0] == index[1]:
                        label[index[0]] = 'S-' + l
                    else:
                        label[index[0]] = 'B-' + l
                        label[index[1]] = 'E-' + l
                        for j in range(index[0] + 1, index[1]):
                            label[j] = 'M-' + l
        return label

    def _create_examples(self, lines):
        """See base class."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s" % (i) if 'id' not in line else line['id']
            text_a = tokenization.convert_to_unicode(line['text'])
            label = ['O'] * len(text_a)
            if 'label' in line:
                label = self.generate_label(line, label)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=[0] * max_seq_length)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in ALBERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens = []
    segment_ids = []
    label_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[example.label[i]])
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)

    input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 1:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %s)" % (example.label, label_ids))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_ids)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a MINDRecord file."""

    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "label_ids": {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0

    for (ex_index, example) in enumerate(examples):
        all_data = []
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        label_ids = np.array(feature.label_id, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    print("Total instances is: ", total_written, flush=True)


def main():
    processors = {
        "ner": NerProcessor
    }

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_file = os.path.join(args.output_dir, "train.mindrecord")
        file_based_convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, train_file)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))

    # ner task with CLUENER do eval with dev.mindrecord
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(args.output_dir, "dev.mindrecord")
        file_based_convert_examples_to_features(eval_examples, label_list,
                                                args.max_seq_length, tokenizer,
                                                eval_file)
        print("***** Running prediction*****")
        print("  Num examples = %d (%d actual, %d padding)",
              len(eval_examples), num_actual_eval_examples,
              len(eval_examples) - num_actual_eval_examples)

    if args.do_predict:
        predict_examples = processor.get_test_examples(args.data_dir)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(args.output_dir, "predict.mindrecord")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                args.max_seq_length, tokenizer,
                                                predict_file)

        print("***** Running prediction*****")
        print("  Num examples = %d (%d actual, %d padding)",
              len(predict_examples), num_actual_predict_examples,
              len(predict_examples) - num_actual_predict_examples)


if __name__ == "__main__":
    args = parse_args()
    main()
