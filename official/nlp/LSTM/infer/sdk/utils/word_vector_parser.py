# coding = utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License  (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import numpy as np


class WordVectorParser:
    """
    parse sentences data to features
    sentence->tokenized->encoded->padding->features
    """

    def __init__(self, sentences_path, vocab_path):
        self.__sentence_dir = sentences_path
        self.__vocab_path = vocab_path

        # properties
        self.__sentence_files = {}
        self.__sentence_datas = {}
        self.__features = {}
        self.__vocab = {}
        self.__word2idx = {}

    def parse(self):
        """
        parse sentences data to memory
        """
        self.__read_sentences_data()
        self.__parse_features()

    def __read_sentences_data(self):
        """
        load data from txt
        """
        self.__sentence_files = os.listdir(self.__sentence_dir)

        data_lists = []
        for file in self.__sentence_files:
            with open(os.path.join(self.__sentence_dir, file), mode='r', encoding='utf8') as f:
                sentence = f.read().replace('\n', '')
                data_lists.append(sentence)
        self.__sentence_datas = data_lists

    def __parse_features(self):
        """
        parse features
        """
        features = []
        for sentence in self.__sentence_datas:
            features.append(sentence)

        self.__features = features

        # update feature to tokenized
        self.__updata_features_to_tokenized()
        # parse vocab
        self.__parse_vocab()
        # encode feature
        self.__encode_features()
        # padding feature
        self.__padding_features()

    def __updata_features_to_tokenized(self):
        """
        split sentence to words
        """
        tokenized_features = []
        for sentence in self.__features:
            tokenized_sentence = [word.lower() for word in sentence.split(" ")]
            tokenized_features.append(tokenized_sentence)
        self.__features = tokenized_features

    def __parse_vocab(self):
        """
        load vocab and generate word indexes
        """
        vocab = []
        with open(self.__vocab_path, 'r') as vocab_file:
            line = vocab_file.readline()
            while line:
                line = line.replace('\n', '')
                vocab.append(line)
                line = vocab_file.readline()
        self.__vocab = vocab

        word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        self.__word2idx = word_to_idx

    def __encode_features(self):
        """
        encode word to index
        """
        word_to_idx = self.__word2idx
        encoded_features = []
        for tokenized_sentence in self.__features:
            encoded_sentence = []
            for word in tokenized_sentence:
                encoded_sentence.append(word_to_idx.get(word, 0))
            encoded_features.append(encoded_sentence)
        self.__features = encoded_features

    def __padding_features(self, max_len=500, pad=0):
        """
        pad all features to the same length
        """
        padded_features = []
        for feature in self.__features:
            if len(feature) >= max_len:
                padded_feature = feature[:max_len]
            else:
                padded_feature = feature
                while len(padded_feature) < max_len:
                    padded_feature.append(pad)
            padded_features.append(padded_feature)
        self.__features = padded_features

    def get_datas(self):
        """
        get features
        """
        features = np.array(self.__features).astype(np.int32)
        files = self.__sentence_files
        return features, files
