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
import collections
import six
import numpy as np
from squad_twomemory import DataProcessor as SquadDataProcessor
from tokenization import FullTokenizer


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
            self.label_map = 0
        else:
            self.label_map = None

    def _read_tsv(self, input_file):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf8") as f:
            reader = csv_reader(f, delimiter="\t")
            headers = next(reader)
            Example = collections.namedtuple('Example', headers)

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

    def get_num_examples(self, input_file):
        """return total number of examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def get_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return examples

    def get_all_path(self, output_path):
        """
        Args:
            output_path: save path of convert dataset
        Returns:
            the path of ids, mask, token, label
        """
        input_mask_path = os.path.join(output_path, "00_data")
        src_ids_path = os.path.join(output_path, "01_data")
        pos_ids_path = os.path.join(output_path, "02_data")
        sent_ids_path = os.path.join(output_path, "03_data")
        wn_concept_ids_path = os.path.join(output_path, "04_data")
        nell_concept_ids_path = os.path.join(output_path, "05_data")
        unique_id_path = os.path.join(output_path, "06_data")
        for path in [input_mask_path, src_ids_path, pos_ids_path, sent_ids_path,
                     wn_concept_ids_path, nell_concept_ids_path, unique_id_path]:
            os.makedirs(path, 0o755, exist_ok=True)

        return input_mask_path, src_ids_path, pos_ids_path, sent_ids_path,\
            wn_concept_ids_path, nell_concept_ids_path, unique_id_path

    def read_concept_embedding(self, embedding_path):
        """read concept embedding"""
        fin = open(embedding_path, encoding='utf-8')
        info = [line.strip() for line in fin]
        dim = len(info[0].split(' ')[1:])
        embedding_mat = []
        id2concept, concept2id = [], {}
        # add padding concept into vocab
        id2concept.append('<pad_concept>')
        concept2id['<pad_concept>'] = 0
        embedding_mat.append([0.0 for _ in range(dim)])
        for line in info:
            concept_name = line.split(' ')[0]
            embedding = [float(value_str) for value_str in line.split(' ')[1:]]
            assert len(embedding) == dim and not np.any(np.isnan(embedding))
            embedding_mat.append(embedding)
            concept2id[concept_name] = len(id2concept)
            id2concept.append(concept_name)
        return concept2id

    def file_based_convert_examples_to_features(self, data_url, output_file):
        """"Convert a set of `InputExample`s to a MindDataset file."""

        wn_concept2id = self.read_concept_embedding(data_url + "/KB_embeddings/wn_concept2vec.txt")
        nell_concept2id = self.read_concept_embedding(data_url + "/KB_embeddings/nell_concept2vec.txt")

        processor = SquadDataProcessor(
            vocab_path=data_url + "/cased_L-24_H-1024_A-16/vocab.txt",
            do_lower_case=False,
            max_seq_length=384,
            in_tokens=False,
            doc_stride=128,
            max_query_length=64)

        print("squad predict data process begin")
        eval_concept_settings = {
            'tokenization_path': data_url + '/tokenization_squad/tokens/dev.tokenization.cased.data',
            'wn_concept2id': wn_concept2id,
            'nell_concept2id': nell_concept2id,
            'use_wordnet': True,
            'retrieved_synset_path': data_url + "/retrieve_wordnet/output_squad/retrived_synsets.data",
            'use_nell': True,
            'retrieved_nell_concept_path': data_url + "/retrieve_nell/output_squad/dev.retrieved_nell_concepts.data",
        }
        eval_data_generator = processor.data_generator(
            data_path=data_url + "/SQuAD/dev-v1.1.json",
            batch_size=1,
            phase='predict',
            shuffle=False,
            dev_count=1,
            epoch=1,
            **eval_concept_settings)

        output_input_mask, output_src_ids, output_pos_ids, output_sent_ids,\
            output_wn_concept_ids, output_nell_concept_ids, output_unique_id = self.get_all_path(output_file)
        example_count = 0

        for example in eval_data_generator():
            src_ids = example[0]
            pos_ids = example[1]
            sent_ids = example[2]
            wn_concept_ids = example[3]
            nell_concept_ids = example[4]
            input_mask = example[5]
            unique_id = example[6]

            nell_concept_ids = np.pad(nell_concept_ids, ((0, 0), (0, 0), (0, 3), (0, 0)),
                                      'constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

            file_name = "squad" + "_" + str(example_count) + ".bin"

            input_mask_file_path = os.path.join(output_input_mask, file_name)
            np.array(input_mask, dtype=np.float32).tofile(input_mask_file_path)

            src_ids_file_path = os.path.join(output_src_ids, file_name)
            np.array(src_ids, dtype=np.int64).tofile(src_ids_file_path)

            pos_ids_file_path = os.path.join(output_pos_ids, file_name)
            np.array(pos_ids, dtype=np.int64).tofile(pos_ids_file_path)

            sent_ids_file_path = os.path.join(output_sent_ids, file_name)
            np.array(sent_ids, dtype=np.int64).tofile(sent_ids_file_path)

            wn_concept_ids_file_path = os.path.join(output_wn_concept_ids, file_name)
            np.array(wn_concept_ids, dtype=np.int64).tofile(wn_concept_ids_file_path)

            nell_concept_ids_file_path = os.path.join(output_nell_concept_ids, file_name)
            np.array(nell_concept_ids, dtype=np.int64).tofile(nell_concept_ids_file_path)

            unique_id_file_path = os.path.join(output_unique_id, file_name)
            np.array(unique_id, dtype=np.int64).tofile(unique_id_file_path)

            example_count += 1
            if example_count % 3000 == 0:
                print(example_count)
        print("total example:", example_count)


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
            Example = collections.namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text.replace(' ', '')
                example = Example(*line)
                examples.append(example)
            return examples


def main():
    parser = argparse.ArgumentParser(description="read dataset and save it to bin")
    parser.add_argument("--vocab_file", type=str, default="", help="vocab file")
    parser.add_argument("--label_map_config", type=str, default=None, help="label mapping config file")
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed number")

    parser.add_argument("--data_path", type=str, default="../data", help="the format of infer file is tsv.")
    parser.add_argument("--output_path", type=str, default="./data", help="the path of convert dataset.")

    args_opt = parser.parse_args()
    reader = ClassifyReader(
        vocab_path=args_opt.vocab_file,
        label_map_config=args_opt.label_map_config,
        max_seq_len=args_opt.max_seq_len,
        do_lower_case=args_opt.do_lower_case,
        random_seed=args_opt.random_seed
    )
    reader.file_based_convert_examples_to_features(data_url=args_opt.data_path, output_file=args_opt.output_path)


if __name__ == "__main__":
    main()
