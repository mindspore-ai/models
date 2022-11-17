"""
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

import os
import math
import random
import argparse
import numpy as np


class MovieReview():
    """
    preprocess MovieReview dataset
    """
    def __init__(self, root_dir, maxlen, split):
        """
        input:
            root_dir: the root directory path of the MR dataset
            maxlen: set the max length of the sentence
            split: set the ratio of training set to testing set
            rank: the logic order of the worker
            size: the worker num
        """
        self.path = root_dir
        self.feelMap = {'neg': 0, 'pos': 1}
        self.files = []
        self.doConvert = False
        mypath = os.path.join(self.path, 'input')
        if not os.path.exists(mypath) or not os.path.isdir(mypath):
            print("please check the root_dir!")
            raise ValueError

        # walk through the root_input_dir

        for root, _, filename in os.walk(mypath):
            for each in filename:
                self.files.append(os.path.join(root, each))
            break

        # check whether get two files
        if len(self.files) != 2:
            print("There are {} files in the root_dir".format(len(self.files)))
            raise ValueError

        # begin to read data
        self.word_num = 0
        self.maxlen = 0
        self.minlen = float("inf")
        self.maxlen = float("-inf")
        self.Pos = []
        self.Neg = []
        for filename in self.files:
            self.read_data(filename)

        self.PosNeg = self.Pos + self.Neg
        self.text2vec(maxlen=maxlen)
        self.split_dataset(split=split)
        self.save2file()

    def read_data(self, filePath):
        """
        read text into memory

        input:
            filePath: the path where the data is stored in
        """
        with open(filePath, 'r', encoding='iso-8859-1') as f:
            for sentence in f.readlines():
                sentence = sentence.replace('\n', '')\
                    .replace('"', '')\
                    .replace('\'', '')\
                    .replace('.', '')\
                    .replace(',', '')\
                    .replace('[', '')\
                    .replace(']', '')\
                    .replace('(', '')\
                    .replace(')', '')\
                    .replace(':', '')\
                    .replace('--', '')\
                    .replace('-', '')\
                    .replace('\\', '')\
                    .replace('0', '')\
                    .replace('1', '')\
                    .replace('2', '')\
                    .replace('3', '')\
                    .replace('4', '')\
                    .replace('5', '')\
                    .replace('6', '')\
                    .replace('7', '')\
                    .replace('8', '')\
                    .replace('9', '')\
                    .replace('`', '')\
                    .replace('=', '')\
                    .replace('$', '')\
                    .replace('/', '')\
                    .replace('*', '')\
                    .replace(';', '')\
                    .replace('<b>', '')\
                    .replace('%', '')
                sentence = sentence.split(' ')
                sentence = list(filter(lambda x: x, sentence))
                if sentence:
                    self.word_num += len(sentence)
                    self.maxlen = self.maxlen if self.maxlen >= len(
                        sentence) else len(sentence)
                    self.minlen = self.minlen if self.minlen <= len(
                        sentence) else len(sentence)
                    if 'pos' in filePath:
                        self.Pos.append([sentence, self.feelMap['pos']])
                    else:
                        self.Neg.append([sentence, self.feelMap['neg']])

    def text2vec(self, maxlen):
        """
        convert the sentence into a vector in an int type

        input:
            maxlen: max length of the sentence
        """
        # Vocab = {word : index}
        self.Vocab = dict()

        for SentenceLabel in self.Pos + self.Neg:
            vector = [0] * maxlen
            for index, word in enumerate(SentenceLabel[0]):
                if index >= maxlen:
                    break
                if word not in self.Vocab.keys():
                    self.Vocab[word] = len(self.Vocab)
                    vector[index] = len(self.Vocab) - 1
                else:
                    vector[index] = self.Vocab[word]
            SentenceLabel[0] = vector
        self.doConvert = True

    def split_dataset(self, split):
        """
        split the dataset into training set and test set
        input:
            split: the ratio of training set to test set
            rank: logic order
            size: device num
        """
        trunk_pos_size = math.ceil((1 - split) * len(self.Pos))
        trunk_neg_size = math.ceil((1 - split) * len(self.Neg))
        trunk_num = int(1 / (1 - split))
        pos_temp = list()
        neg_temp = list()
        for index in range(trunk_num):
            pos_temp.append(self.Pos[index * trunk_pos_size:(index + 1) *
                                     trunk_pos_size])
            neg_temp.append(self.Neg[index * trunk_neg_size:(index + 1) *
                                     trunk_neg_size])
        self.test = pos_temp.pop(2) + neg_temp.pop(2)
        # self.train = [i for item in pos_temp + neg_temp for i in item]
        self.train = []
        for item in pos_temp + neg_temp:
            for i in item:
                self.train.append(i)

        random.shuffle(self.train)
        random.shuffle(self.test)

    def save2file(self):
        """
        save the serialized sentences into binary files prepared for  test and train process
        """
        ids_input = []
        labels_input = []
        ids_path = os.path.join(self.path, 'ids')
        if not os.path.exists(ids_path):
            os.makedirs(ids_path)
        labels_path = os.path.join(self.path, 'labels')
        if not os.path.exists(labels_path):
            os.makedirs(labels_path)
        ids_total = len(self.test)
        for i in range(ids_total):
            ids_input = self.test[i][0]
            labels_input = self.test[i][1]
            file_name = "ids/" + str(i) + ".bin"
            file_path = os.path.join(self.path, file_name)
            np.array(ids_input, dtype=np.int32).tofile(file_path)
            file_name = "labels/" + str(i) + ".bin"
            file_path = os.path.join(self.path, file_name)
            np.array(labels_input, dtype=np.int32).tofile(file_path)
        print("\n               ******         Success!        ******\n  ")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TextCNN Dataset Pre_process')
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=51,
                        help="sentence length, default is 51.")
    parser.add_argument("--data_path",
                        type=str,
                        default="../data",
                        help="the path of convert dataset.")
    args_opt = parser.parse_args()

    instance = MovieReview(root_dir=args_opt.data_path,
                           maxlen=args_opt.max_seq_len,
                           split=0.9)
