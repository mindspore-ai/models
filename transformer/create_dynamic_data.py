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
"""Create training instances for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import src.tokenization as tokenization
from src.model_utils.config import config
from mindspore.mindrecord import FileWriter

class SampleInstance():
    """A single sample instance (sentence pair)."""

    def __init__(self, source_sos_tokens, source_eos_tokens, target_sos_tokens, target_eos_tokens):
        self.source_sos_tokens = source_sos_tokens
        self.source_eos_tokens = source_eos_tokens
        self.target_sos_tokens = target_sos_tokens
        self.target_eos_tokens = target_eos_tokens

    def __str__(self):
        s = ""
        s += "source sos tokens: %s\n" % (" ".join(
            [tokenization.convert_to_printable(x) for x in self.source_sos_tokens]))
        s += "source eos tokens: %s\n" % (" ".join(
            [tokenization.convert_to_printable(x) for x in self.source_eos_tokens]))
        s += "target sos tokens: %s\n" % (" ".join(
            [tokenization.convert_to_printable(x) for x in self.target_sos_tokens]))
        s += "target eos tokens: %s\n" % (" ".join(
            [tokenization.convert_to_printable(x) for x in self.target_eos_tokens]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def get_instance_features(instance, tokenizer, max_seq_length):
    """Get features from `SampleInstance`s."""
    def _convert_ids_and_mask(input_tokens, max_seq_length):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

    source_eos_ids, source_eos_mask = _convert_ids_and_mask(instance.source_eos_tokens, max_seq_length)
    target_sos_ids, target_sos_mask = _convert_ids_and_mask(instance.target_sos_tokens, max_seq_length)
    target_eos_ids, target_eos_mask = _convert_ids_and_mask(instance.target_eos_tokens, max_seq_length)

    return np.asarray(source_eos_ids), np.asarray(source_eos_mask), np.asarray(target_sos_ids), \
      np.asarray(target_sos_mask), np.asarray(target_eos_ids), np.asarray(target_eos_mask)


def create_training_instance(source_words, target_words, max_seq_length, clip_to_max_len):
    """Creates `SampleInstance`s for a single sentence pair."""
    EOS = "</s>"
    SOS = "<s>"

    if len(source_words) >= max_seq_length or len(target_words) >= max_seq_length:
        if clip_to_max_len:
            source_words = source_words[:min([len(source_words, max_seq_length-1)])]
            target_words = target_words[:min([len(target_words, max_seq_length-1)])]
        else:
            return None

    source_sos_tokens = [SOS] + source_words
    source_eos_tokens = source_words + [EOS]
    target_sos_tokens = [SOS] + target_words
    target_eos_tokens = target_words + [EOS]

    instance = SampleInstance(
        source_sos_tokens=source_sos_tokens,
        source_eos_tokens=source_eos_tokens,
        target_sos_tokens=target_sos_tokens,
        target_eos_tokens=target_eos_tokens)
    return instance


def align_down(n, align):
    return n & ~(align - 1)


def read_data_from_file(tokenizer, input_files, all_data):
    """
    read data from file.
    """
    for input_file in input_files:
        logging.info("*** Reading from   %s ***", input_file)
        with open(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break

                source_line, target_line = line.strip().split("\t")
                source_tokens = tokenizer.tokenize(source_line)
                target_tokens = tokenizer.tokenize(target_line)

                if len(source_tokens) >= config.max_seq_length or len(target_tokens) >= config.max_seq_length:
                    logging.info("ignore long sentence!")
                    continue

                tmp_len = len(source_tokens) if len(source_tokens) > len(target_tokens) else len(target_tokens)
                one_line = (source_tokens, target_tokens, tmp_len)
                all_data.append(one_line)


def config_data(tokenizer, all_data, all_batches):
    """
    config mindrecord data.
    """
    index = config.batch_size - 1
    max_length = all_data[index][2]

    batch = []
    list_source_eos_ids = []
    list_source_eos_mask = []
    list_target_sos_ids = []
    list_target_sos_mask = []
    list_target_eos_ids = []
    list_target_eos_mask = []

    all_count = align_down(len(all_data), config.batch_size)
    i = 0

    for data in all_data:
        if i == all_count:
            logging.info("align down data length to %d.", all_count)
            break
        instance = create_training_instance(data[0], data[1], config.max_seq_length,
                                            clip_to_max_len=config.clip_to_max_len)

        if instance is None:
            continue

        source_eos_ids, source_eos_mask, target_sos_ids, target_sos_mask, target_eos_ids, target_eos_mask \
        = get_instance_features(instance, tokenizer, max_length + 1)

        list_source_eos_ids.append(source_eos_ids)
        list_source_eos_mask.append(source_eos_mask)
        list_target_sos_ids.append(target_sos_ids)
        list_target_sos_mask.append(target_sos_mask)
        list_target_eos_ids.append(target_eos_ids)
        list_target_eos_mask.append(target_eos_mask)

        index = index + 1
        i = i + 1

        if len(list_source_eos_ids) == config.batch_size and index < len(all_data):
            batch.append(list_source_eos_ids)
            batch.append(list_source_eos_mask)
            batch.append(list_target_sos_ids)
            batch.append(list_target_sos_mask)
            batch.append(list_target_eos_ids)
            batch.append(list_target_eos_mask)

            col_num = len(batch)
            batch_shape = (col_num, config.batch_size, max_length + 1)
            batch_info = (np.array(batch).reshape(-1), batch_shape)
            all_batches.append(batch_info)

            max_length = all_data[index][2]

            batch = []
            list_source_eos_ids = []
            list_source_eos_mask = []
            list_target_sos_ids = []
            list_target_sos_mask = []
            list_target_eos_ids = []
            list_target_eos_mask = []


def write_data(all_batches):
    output_file_name = ""
    data_schema = {"batch_data": {"type": "int64", "shape": [-1]},
                   "batch_shape": {"type": "int64", "shape": [-1]},}
    output_file_name = config.output_file
    writer = FileWriter(output_file_name, 1)
    writer.add_schema(data_schema, "transformer")
    for batch_info in all_batches:
        writer.write_raw_data([{"batch_data": np.array(batch_info[0]), "batch_shape": np.array(batch_info[1])}])
    writer.commit()


def main():
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=config.vocab_file)

    input_files = []
    for input_pattern in config.input_file.split(","):
        input_files.append(input_pattern)

    logging.info("*** Read from input files ***")
    for input_file in input_files:
        logging.info("  %s", input_file)

    output_file = config.output_file
    logging.info("*** Write to output files ***")
    logging.info("  %s", output_file)

    all_data = []
    read_data_from_file(tokenizer, input_files, all_data)

    all_data = sorted(all_data, key=lambda x: x[2])

    all_batches = []
    logging.info("start config data")
    config_data(tokenizer, all_data, all_batches)
    logging.info("finish config data")

    logging.info("start write database")
    write_data(all_batches)
    logging.info("finish write database")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
