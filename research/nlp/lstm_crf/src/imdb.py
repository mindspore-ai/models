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
"""
imdb dataset parser.
"""
import os
import numpy as np
from .model_utils.config import config


UNK = "<UNK>"
PAD = "<PAD>"
NUM = "<NUM>"


def modelarts_pre_process():
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
            ERROR: Unable to locate file {}.
            FIX: Have you tried running run_build_data.sh first?
            This will build vocab file from your train and test sets and
            your word vectors.
            """.format(filename)
        super(MyIOError, self).__init__(message)


class ImdbParser():
    """
    parse data to features and labels.
    sentence->tokenized->encoded->padding->features
    """

    def __init__(self, imdb_path, glove_path, words_path, embed_size=300):
        self.__imdb_path = imdb_path
        self.__glove_dim = embed_size
        self.__glove_file = os.path.join(glove_path, 'glove.6B.' + str(self.__glove_dim) + 'd.txt')
        self.__glove_vectors_path = os.path.join(words_path, 'glove.6B.'+ str(self.__glove_dim) + 'trimmed.npz')
        self.__words_path = os.path.join(words_path, 'words.txt')
        self.__tags_path = os.path.join(words_path, 'tags.txt')
        self.__max_length_path = os.path.join(words_path, 'max_length.txt')
        # properties
        self.words = []
        self.tags = []
        self.glove_vocab = set()
        self.vocab_words = set()
        self.vocab_tags = set()
        self.vocab_dict = dict()
        self.words_to_index_map = {}
        self.tags_to_index_map = {}
        self.words_index = []
        self.tags_index = []
        self.sequence_pad = []
        self.sequence_tag_pad = []

    def __get_words_tags(self, seg='train', build_data=True):
        '''load data from txt'''

        if build_data:
            segs = ['train', 'test']
        else:
            segs = seg
            print('segs:', segs)
        for i in segs:
            sentence_dir = os.path.join(self.__imdb_path, i) + '.txt'
            print('load....', sentence_dir, 'data')
            with open(sentence_dir, mode='r', encoding='utf-8') as f:
                word_list = []
                tag_list = []
                for line in f:
                    if line != '\n':
                        word, _, tag = line.strip('\n').split()
                        word = word.lower()
                        if word.isdigit():
                            word = NUM
                        word_list.append(word)
                        tag_list.append(tag)
                    else:
                        self.words.append(word_list)
                        self.tags.append(tag_list)
                        word_list = []
                        tag_list = []
        self.max_length = max([len(self.words[i]) for i in range(len(self.words))])

    def __write_max_sequence_length(self):
        with open(self.__max_length_path, 'w') as f:
            f.write(str(self.max_length))

    def __get_vocabs(self):
        for i in range(len(self.words)):
            self.vocab_words.update(self.words[i])
            self.vocab_tags.update(self.tags[i])

    def __get_glove_vocab(self):
        with open(self.__glove_file, mode='r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split(' ')[0]
                self.glove_vocab.add(word)

    def __write_vocab(self, path, data):
        print("Writing vocab......")
        with open(path, "w") as f:
            for i, word in enumerate(data):
                if i != len(data)-1:
                    f.write('{}\n'.format(word))
                else:
                    f.write(word)
        print("- done. {} tokens".format(len(data)))

    def __get_glove_vectors(self):
        """embedding"""
        embeddings = np.zeros([len(self.vocab), self.__glove_dim])
        with open(self.__glove_file, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in self.words_to_index_map:
                    word_index = self.words_to_index_map[word]
                    embeddings[word_index] = np.asarray(embedding)
        np.savez_compressed(self.__glove_vectors_path, embeddings=embeddings)

    def __get_pretrain_glove_vectors(self):
        try:
            with np.load(self.__glove_vectors_path) as data:
                return data["embeddings"]
        except IOError:
            raise MyIOError(self.__glove_vectors_path)

    def __load_vocab_map(self, path, datas_var):
        vocab_map = datas_var
        try:
            with open(path, encoding='utf-8') as f:
                for index, word in enumerate(f):
                    word = word.strip()
                    vocab_map[word] = index

        except IOError:
            raise MyIOError(path)

    def __get_sequence_length(self):
        with open(self.__max_length_path, encoding='utf-8') as f:
            for word in f:
                self.sequence_max_length = int(word)

    def __get_vocab_index(self, data_list, path_map, path_index):
        """glove vector"""
        for words in data_list:
            vocab_index = []
            for word in words:
                if word in path_map:
                    vocab_index.append(path_map[word])
                else:
                    vocab_index.append(path_map[UNK])
            path_index.append(vocab_index)

    def __get_sequence_same_length(self):
        for vocab_index in self.words_index:
            vocab_index_ = vocab_index[:self.sequence_max_length] + \
                [0]*max((self.sequence_max_length-len(vocab_index)), 0)
            self.sequence_pad += [vocab_index_]

    def __get_sequence_tag_same_length(self):
        for vocab_index in self.tags_index:
            vocab_index_ = vocab_index[:self.sequence_max_length] + \
                [0]*max((self.sequence_max_length-len(vocab_index)), 0)
            self.sequence_tag_pad += [vocab_index_]

    def build_datas(self, seg, build_data):
        """build the vocab and embedding"""
        self.__get_words_tags(seg, build_data)
        self.__get_vocabs()
        self.__get_glove_vocab()
        self.vocab = self.vocab_words & self.glove_vocab
        self.vocab.add(UNK)
        self.vocab.add(NUM)
        self.vocab.add('<START>')
        self.vocab.add('<STOP>')
        self.vocab = list(self.vocab)
        self.vocab.insert(0, PAD)
        self.__write_vocab(self.__words_path, self.vocab)
        self.__write_vocab(self.__tags_path, self.vocab_tags)
        self.__write_max_sequence_length()
        self.__load_vocab_map(self.__words_path, self.words_to_index_map)
        self.__load_vocab_map(self.__tags_path, self.tags_to_index_map)
        self.__get_glove_vectors()

    def get_datas_embeddings(self, seg, build_data):
        """read the CoNLL2000 data embedding"""
        self.__get_words_tags(seg, build_data)
        embeddings = self.__get_pretrain_glove_vectors()
        self.__load_vocab_map(self.__words_path, self.words_to_index_map)
        self.__load_vocab_map(self.__tags_path, self.tags_to_index_map)
        self.__get_vocab_index(self.words, self.words_to_index_map, self.words_index)
        self.__get_vocab_index(self.tags, self.tags_to_index_map, self.tags_index)
        self.__get_sequence_length()
        self.__get_sequence_same_length()
        self.__get_sequence_tag_same_length()
        return embeddings, self.sequence_max_length, self.words, self.tags, self.sequence_pad, \
               self.sequence_tag_pad, self.tags_to_index_map
