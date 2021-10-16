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
"""Change cor Dataset to bin file"""
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Create AttentionLSTM dataset.')
parser.add_argument("--data_folder", type=str, required="./data", help="data folder.")
parser.add_argument("--glove_file", type=str, default="./glove.840B.300d.txt", help="glove 300d file.")
parser.add_argument("--result_path", type=str, default="", help="result folder")
args, _ = parser.parse_known_args()


class Sentence():
    """docstring for sentence"""
    def __init__(self, content, target, rating, grained):
        self.content, self.target = content.lower(), target
        self.solution = np.zeros(grained, dtype=np.float32)
        self.senlength = len(self.content.split(' '))
        try:
            self.solution[int(rating)+1] = 1
        except SystemExit:
            exit()

    def stat(self, target_dict, wordlist, grained=3):
        """statistical"""
        data, data_target, i = [], [], 0
        solution = np.zeros((self.senlength, grained), dtype=np.float32)
        for word in self.content.split(' '):
            data.append(wordlist[word])
            try:
                pol = Lexicons_dict[word]
                solution[i][pol + 1] = 1
            except NameError:
                pass
            i = i + 1
        for word in self.target.split(' '):
            data_target.append(wordlist[word])
        return {'seqs': data,
                'target': data_target,
                'solution': np.array([self.solution]),
                'target_index': self.get_target(target_dict)}

    def get_target(self, dict_target):
        """
        target
        """
        return dict_target[self.target]


class DataManager():
    """create bin dataset"""
    def __init__(self, dataset, grained=3):
        self.fileList = ['train', 'test', 'dev']
        self.origin = {}
        self.wordlist = {}
        self.data = {}
        for fname in self.fileList:
            data = []
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in range(int(len(sentences)/3)):
                    content, target = sentences[i * 3].strip(), sentences[i * 3 + 1].strip()
                    rating = sentences[i * 3 + 2].strip()
                    sentence = Sentence(content, target, rating, grained)
                    data.append(sentence)
            self.origin[fname] = data
        self.gen_target()

    def gen_word(self):
        """Statistical characters"""
        wordcount = {}
        def sta(sentence):
            """
            Sentence Statistical
            """
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except LookupError:
                    wordcount[word] = 1
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except LookupError:
                    wordcount[word] = 1

        for fname in self.fileList:
            for sent in self.origin[fname]:
                sta(sent)
        words = wordcount.items()
        sorted(words, key=lambda x: x[1], reverse=True)
        self.wordlist = {item[0]: index + 1 for index, item in enumerate(words)}
        return self.wordlist

    def gen_target(self, threshold=5):
        """Statistical aspect"""
        self.dict_target = {}
        for fname in self.fileList:
            for sent in self.origin[fname]:
                if sent.target in self.dict_target:
                    self.dict_target[sent.target] = self.dict_target[sent.target] + 1
                else:
                    self.dict_target[sent.target] = 1
        i = 0
        for (key, val) in self.dict_target.items():
            if val < threshold:
                self.dict_target[key] = 0
            else:
                self.dict_target[key] = i
                i = i + 1
        return self.dict_target

    def gen_data(self, grained=3):
        """all data"""
        if grained != 3:
            print("only support 3")

        for fname in self.fileList:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_target, self.wordlist))
        return self.data['train'], self.data['dev'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        """word to vector"""
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if tmp[0] in mdict:
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted)-1) + ' ' + str(len(line.strip().split())-1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))


def _convert_to_bin(data):
    """
    convert cor dataset to bin dataset
    """
    print("convert to bin files...")

    content = []
    sen_len = []
    aspect = []
    solution = []
    aspect_index = []

    for info in data:
        content.append(info['seqs'])
        aspect.append(info['target'])
        sen_len.append([len(info['seqs'])])
        solution.append(info['solution'])
        aspect_index.append(info['target_index'])

    padded_content = np.zeros([len(content), 50])
    for index, seq in enumerate(content):
        if len(seq) <= 50:
            padded_content[index, 0:len(seq)] = seq
        else:
            padded_content[index] = seq[0:50]

    content = padded_content

    content_path = os.path.join(args.result_path, "00_content")
    sen_len_path = os.path.join(args.result_path, "01_sen_len")
    aspect_path = os.path.join(args.result_path, "02_aspect")
    solution_path = os.path.join(args.result_path, "solution_path")
    if not os.path.isdir(content_path):
        os.makedirs(content_path)
    if not os.path.isdir(sen_len_path):
        os.makedirs(sen_len_path)
    if not os.path.isdir(aspect_path):
        os.makedirs(aspect_path)
    if not os.path.isdir(solution_path):
        os.makedirs(solution_path)
    for i, _ in enumerate(content):
        file_name = "atae_lstm_bs1" + "_" + str(i) + ".bin"
        content_bin = np.array(content[i]).astype(np.int32)
        content_bin.tofile(os.path.join(content_path, file_name))
        sen_len_bin = np.array(int(sen_len[i][0])).astype(np.int32)
        sen_len_bin.tofile(os.path.join(sen_len_path, file_name))
        aspect_bin = np.array(int(aspect[i][0])).astype(np.int32)
        aspect_bin.tofile(os.path.join(aspect_path, file_name))
        solution_bin = np.array(solution[i][0]).astype(np.int32)
        solution_bin.tofile(os.path.join(solution_path, file_name))

    print("=" * 10, "export bin files finished", "=" * 10)


def wordlist_to_glove_weight(wordlist, glove_file):
    """load glove word vector"""
    glove_word_dict = {}
    with open(glove_file) as f:
        line = f.readline()
        while line:
            array = line.split(' ')
            word = array[0]
            glove_word_dict[word] = array[1:301]
            line = f.readline()

    weight = np.zeros((len(wordlist)+1, 300)).astype(np.float32)+0.01
    unfound_count = 0
    for word, i in wordlist.items():
        word = word.strip()
        if word in glove_word_dict:
            weight[i] = glove_word_dict[word]
        else:
            unfound_count += 1

    print("not found in glove: ", unfound_count)
    print(np.shape(weight))
    print(weight.dtype)

    np.savez('./weight.npz', weight=weight)


if __name__ == "__main__":
    # data pth
    train_data_path = os.path.join(args.data_folder, 'train.cor')
    test_data_path = os.path.join(args.data_folder, 'test.cor')
    dev_data_path = os.path.join(args.data_folder, 'dev.cor')

    data_all = DataManager(args.data_folder)
    word_list = data_all.gen_word()
    print("word_list: ", type(word_list))

    wordlist_to_glove_weight(word_list, args.glove_file)

    train_data, dev_data, test_data = data_all.gen_data(grained=3)

    _convert_to_bin(test_data)
