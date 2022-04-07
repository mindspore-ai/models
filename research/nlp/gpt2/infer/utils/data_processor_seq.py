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

"""create data for LM task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import logging
from typing import List, Optional
import numpy as np
import regex as re

logger = logging.getLogger(__name__)

def bytes_to_unicode():
    """
    bytes to unicode
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(i) for i in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class GPT2Tokenizer():
    """
    GPT2Tokenizer
    """
    def __init__(
            self,
            vocab_file,
            merge_file,
            add_prefix_space=False,
    ):
        with open(vocab_file, 'r', encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.decoder)
        with open(merge_file, 'r', encoding="utf-8") as merge_handle:
            bpe_merges = merge_handle.read().split('\n')[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.add_prefix_space = add_prefix_space
        self.cache = {}

        self.unk_token = "<|endoftext|>"
        self.unk_token_id = 50256
        self.bos_token = "<|endoftext|>"
        self.bos_token_id = 50256
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 50256
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = 50256

    def bpe(self, token):
        """
        bpe encode
        """

        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(token)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """ Tokenize a string using bpe encode. """
        text = self.prepare_for_tokenization(text, is_pretokenized=False)
        # print(text)
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """ the index of the token in the vocabulary. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, _id):
        """ return the origin bpe token according to id"""
        return self.decoder.get(_id)

    def _convert_tokens_to_string(self, tokens):
        """ return a string according to the list of tokens"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors='ignore')
        return text

    def encode(self, text):
        """ get the index list of text"""
        text_id = []
        bpe_tokens = self._tokenize(text)
        for token in bpe_tokens:
            text_id.append(self._convert_token_to_id(token))
        return text_id

    def decode(self, ids):
        """ return a string according to the index list of tokens"""
        tokens = []
        for id_ in ids:
            tokens.append(self._convert_id_to_token(id_))
        return self._convert_tokens_to_string(tokens)

    def prepare_for_tokenization(self, text, is_pretokenized=False, **kwargs):
        """ whether to add a whitespace in the front of text """
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_pretokenized or add_prefix_space:
            text = " " + text
        return text

    def num_special_tokens_to_add(self, pair: bool = False):
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence by concatenating and adding special tokens.

        A GPT2 sequence has the following format:
        - single sequence: ``<bos> X <eos>``
        - pair of sequences: ``<bos> A <eos> B <eos>``

        Args:
            token_ids_0 (List[int]): List of IDs to which the special tokens will be added
            token_ids_1 (List[int], `optional`, defaults to `None`): Optional second list of IDs for sequence pairs.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def truncate_sequences(self, ids, num_tokens_to_remove, truncation_strategy="ONLY_FIRST", direction="RIGHT"):
        """
        truncate sequences
        Args:
            ids: Any
            num_tokens_to_remove:
            truncation_strategy: str
            direction: str

        Returns:
            (ids, overflowing_tokens): (Any, list)

        """
        if num_tokens_to_remove <= 0:
            return ids, []

        overflowing_tokens = []
        if truncation_strategy == "ONLY_FIRST":
            if len(ids) > num_tokens_to_remove:
                if direction == "RIGHT":
                    overflowing_tokens = ids[-num_tokens_to_remove:]
                    ids = ids[:-num_tokens_to_remove]
                if direction == "LEFT":
                    overflowing_tokens = ids[:num_tokens_to_remove]
                    ids = ids[num_tokens_to_remove:]
            else:
                logger.error("The first sequence length is smaller than removed tokens. ")
        else:
            logger.error("Please select correct truncation strategy, for instance 'ONLY_FIRST'")
        return (ids, overflowing_tokens)

    def _pad(self, encoded_inputs, max_length=None, padding_strategy=None,
             return_attention_mask: Optional[bool] = None):
        """
        _pad
        Args:
            encoded_inputs:
            max_length: Any
            padding_strategy: Any
            return_attention_mask: Optional[bool]

        Returns:
            encoded_inputs:

        """
        needs_to_be_padded = (len(encoded_inputs["input_ids"]) != max_length)
        if needs_to_be_padded:
            if padding_strategy == "MAX_LENGTH":
                difference = max_length - len(encoded_inputs["input_ids"])
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                    encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            else:
                raise ValueError("Invalid padding strategy")
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        return encoded_inputs

    def pad(self, encoded_inputs, max_length: Optional[int] = None, padding_strategy="MAX_LENGTH",
            return_attention_mask=True):
        """
        pad
        Args:
            encoded_inputs:
            max_length: Optional[int]
            padding_strategy: str
            return_attention_mask: bool

        Returns:
            batch_outputs: Dict[Any, list]

        """
        # no batch encoded_inputs["input_ids"]--->[98, 67, 32388, 318, 1912, 287, 170, 8496, 318, 905, 2667, 32]
        if encoded_inputs["input_ids"] and not isinstance(encoded_inputs["input_ids"][0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask
            )
            return encoded_inputs

        # encoded_inputs with batch_size
        batch_size = len(encoded_inputs["input_ids"])
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == "LONGEST":
            max_length = max(len(inputs) for inputs in encoded_inputs["input_ids"])
            padding_strategy = "MAX_LENGTH"

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                encoded_inputs=inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask
            )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return batch_outputs

    def prepare_for_model(self,
                          ids,
                          pair_ids=None,
                          add_special_tokens=True,
                          max_length=None,
                          padding=None,
                          truncate_direction="RIGHT",
                          return_overflowing_tokens=False,
                          return_attention_mask=True):
        """
        prepare for model
        Args:
            ids:
            pair_ids:
            add_special_tokens: bool
            max_length: Any
            padding: Any
            truncate_direction: str
            return_overflowing_tokens: bool
            return_attention_mask: bool

        Returns:
            encoded_inputs:Dict

        """

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        if max_length and total_len > max_length:

            ids, overflowing_tokens = self.truncate_sequences(ids=ids,
                                                              num_tokens_to_remove=total_len - max_length,
                                                              truncation_strategy="ONLY_FIRST",
                                                              direction=truncate_direction)
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids

        # build output dictionary
        encoded_inputs["input_ids"] = sequence
        # check lengths
        if max_length is None or len(encoded_inputs["input_ids"]) > max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model (%ids > %length). Running this sequence through the model will result in "
                "indexing errors", len(ids), max_length
            )
        # padding
        if padding or return_attention_mask:
            encoded_inputs = self.pad(encoded_inputs=encoded_inputs,
                                      max_length=max_length,
                                      padding_strategy="MAX_LENGTH",
                                      return_attention_mask=return_attention_mask)

        return encoded_inputs

def create_instance(tokenizer, text, max_length=None):
    """A single sample instance for LM task."""
    sentence = text.strip().split("\t")

    ids = tokenizer.encode(sentence[0])
    pair_ids = None
    if len(sentence) == 2:
        pair_ids = tokenizer.encode(sentence[1])

    output = tokenizer.prepare_for_model(ids=ids,
                                         pair_ids=pair_ids,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         padding=True,
                                         truncate_direction="LEFT",
                                         return_overflowing_tokens=False,
                                         return_attention_mask=True)
    return output

def write_instance_to_file(instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    input_mask = instance["attention_mask"]
    label_ids = instance["input_ids"]
    assert len(input_ids) == len(label_ids)

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids)
    features["input_mask"] = np.asarray(input_mask)
    features["label_ids"] = np.asarray(label_ids)

    return features

def dataset_preprocess_seq(input_file, tokenizer, max_length):
    """ infer dataset preprocess for PTB or 1BW"""
    print("***** Reading from  %s *****", input_file)
    input_ids = []
    input_mask = []
    label_ids = []
    total_read = 0
    total_written = 0
    with open(input_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            total_read += 1
            if total_read % 500 == 0:
                print("%d ...", total_read)

            output = create_instance(tokenizer, line, max_length)
            features = write_instance_to_file(instance=output)
            total_written += 1

            if total_written <= 20:
                print("***** Example *****")
                print("input tokens: %s", tokenizer.decode(output["input_ids"][:-1]))
                print("label tokens: %s", tokenizer.decode(output["input_ids"][1:]))

                input_ids.append(np.array(features['input_ids'], dtype=np.int64))
                input_mask.append(np.array(features['input_mask'], dtype=np.int64))
                label_ids.append(np.array(features['label_ids'], dtype=np.int64))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    print("%s: %s", feature_name, feature)
    print("Wrote %d total instances", total_written)
    return input_ids, input_mask, label_ids

def wikitext_dataset_preprocess_seq(input_file, tokenizer, max_length):
    """ infer dataset preprocess for wikitext"""
    print("***** Reading from  %s *****", input_file)
    passage = []
    input_ids = []
    input_mask = []
    label_ids = []
    total_read = 0
    total_written = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith('=') and line.endswith('=') and passage:
                    passage = []
                elif line.startswith('=') and line.endswith('='):
                    continue
                else:
                    passage.append(line)
                    total_read += 1
                    if total_read % 500 == 0:
                        print("%d ...", total_read)

                    output = create_instance(tokenizer, line, max_length)
                    features = write_instance_to_file(instance=output)
                    total_written += 1

                    if total_written <= 20:
                        print("***** Example *****")
                        print("input tokens: %s", tokenizer.decode(output["input_ids"][:-1]))
                        print("label tokens: %s", tokenizer.decode(output["input_ids"][1:]))

                        input_ids.append(np.array(features['input_ids'], dtype=np.int64))
                        input_mask.append(np.array(features['input_mask'], dtype=np.int64))
                        label_ids.append(np.array(features['label_ids'], dtype=np.int64))

                        for feature_name in features.keys():
                            feature = features[feature_name]
                            print("%s: %s", feature_name, feature)
    print("Wrote %d total instances", total_written)
    return input_ids, input_mask, label_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, required=True, help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help='The MindRecord file will be split into the number of partition. ')
    parser.add_argument("--max_length", type=int, required=True, help='Maximum sequence length. ')
    parser.add_argument("--dataset", type=str, default="ptb",
                        help="The name of dataset which should be processed, only for LanguageModeling task.")
    parser.add_argument("--vocab_file", type=str, required=True, default='', help='url of gpt2-vocab.json ')
    parser.add_argument("--merge_file", type=str, required=True, default='', help='url of gpt2-merges.txt ')
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merge_file=args.merge_file)

    input_file = args.input_file
    print("***** Reading from input files *****")
    print("Input File: %s", input_file)

    output_file = args.output_file
    print("***** Writing to output files *****")
    print("Output File: %s", output_file)

    if args.dataset == "ptb" or args.dataset == "onebw":
        input_ids, input_mask, label_ids = dataset_preprocess_seq(input_file,
                                                                  tokenizer, args.max_length)
    elif args.dataset == "wikitext2" or args.dataset == "wikitext103":
        input_ids, input_mask, label_ids = wikitext_dataset_preprocess_seq(input_file,
                                                                           tokenizer, args.max_length)
    else:
        raise Exception("Dataset not supported. support: [ptb, onebw, wikitext2, wikitext103]")

    np.savetxt(output_file + '/input_ids.txt', np.array(input_ids), fmt='%i', delimiter=' ')
    np.savetxt(output_file + '/input_mask.txt', np.array(input_mask), fmt='%i', delimiter=' ')
    np.savetxt(output_file + '/label_ids.txt', np.array(label_ids), fmt='%i', delimiter=' ')

if __name__ == "__main__":
    main()
