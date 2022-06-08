# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""utils"""
import os
import time
import random
from datetime import timedelta
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype

MAX_VOCAB_SIZE = 5000000
UNK, PAD = '<UNK>', '<PAD>'


def hash_str(gram_str):
    """hash fun"""
    gram_bytes = bytes(gram_str, encoding='utf-8')
    hash_size = 18446744073709551616
    h = 2166136261
    for gram in gram_bytes:
        h = h ^ gram
        h = (h * 1677619) % hash_size
    return h


def addWordNgrams(hash_list, n, bucket):
    """add word grams"""
    ngram_hash_list = []
    len_hash_list = len(hash_list)
    for index, hash_val in enumerate(hash_list):
        bound = min(len_hash_list, index + n)

        for i in range(index + 1, bound):
            hash_val = hash_val * 116049371 + hash_list[i]
            ngram_hash_list.append(hash_val % bucket)

    return ngram_hash_list


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """build vocab"""
    vocab_dic = {}
    label_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_splits = line.split("\t")
            if len(line_splits) != 2:
                print(line)
            content, label = line_splits
            label_set.add(label.strip())

            for word in tokenizer(content.strip()):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]

        vocab_list = [[PAD, 111101], [UNK, 111100]] + vocab_list
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}

    base_datapath = os.path.dirname(file_path)
    with open(os.path.join(base_datapath, "vocab.txt"), "w", encoding="utf-8") as f:
        for w, c in vocab_list:
            f.write(str(w) + " " + str(c) + "\n")
        # 增加两个demo
        f.write("4654654#%$#%$#" + " " + str(1) + "\n")
        f.write("46#$%54#%$#%$#" + " " + str(1) + "\n")
    with open(os.path.join(base_datapath, "labels.txt"), "w", encoding="utf-8") as fr:
        labels_list = list(label_set)
        labels_list.sort()
        for l in labels_list:
            fr.write(l + "\n")
    return vocab_dic, list(label_set)


def _pad(data, pad_id, width=-1):
    """pad function"""
    if width == -1:
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def load_vocab(vocab_path, max_size, min_freq):
    """load vocab"""
    vocab = {}
    with open(vocab_path, 'r', encoding="utf-8") as fhr:
        for line in fhr:
            line = line.strip()
            line = line.split(' ')
            if len(line) != 2:
                continue
            token, count = line
            vocab[token] = int(count)

    sorted_tokens = sorted([item for item in vocab.items() if item[1] >= min_freq], key=lambda x: x[1], reverse=True)
    sorted_tokens = sorted_tokens[:max_size]
    all_tokens = [[PAD, 0], [UNK, 0]] + sorted_tokens
    vocab = {item[0]: i for i, item in enumerate(all_tokens)}
    return vocab


def load_labels(label_path):
    """load labels"""
    labels = []
    with open(label_path, 'r', encoding="utf-8") as fhr:
        for line in fhr:
            line = line.strip()
            if line not in labels:
                labels.append(line)
    return labels


def build_dataset(config, use_word, min_freq=5):
    """build dataset"""
    print("use min words freq:%d" % (min_freq))
    if use_word:
        tokenizer = lambda x: x.split(' ')  # word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    _ = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=min_freq)

    vocab = load_vocab(config.vocab_path, max_size=MAX_VOCAB_SIZE, min_freq=min_freq)
    print("Vocab size:", len(vocab))
    labels = load_labels(config.labels_path)
    print("label size:", len(labels))

    train = TextDataset(
        file_path=config.train_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        device=config.device,
        max_length=config.max_length,
        nraws=80000,
        shuffle=True
    )

    dev = TextDataset(
        file_path=config.dev_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        device=config.device,
        max_length=config.max_length,
        nraws=80000,
        shuffle=False
    )

    test = TextDataset(
        file_path=config.test_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        device=config.device,
        max_length=config.max_length,
        nraws=80000,
        shuffle=False
    )

    config.class_list = labels
    config.num_classes = len(labels)

    return vocab, train, dev, test


class TextDataset:
    """textdataset struct"""

    def __init__(self, file_path, vocab, labels, tokenizer, wordNgrams,
                 buckets, device, max_length=32, nraws=80000, shuffle=False):

        file_raws = 0
        with open(file_path, 'r', encoding="utf-8") as f:
            for _ in f:
                file_raws += 1
        self.file_path = file_path
        self.file_raws = file_raws
        if file_raws < 200000:
            self.nraws = file_raws
        else:
            self.nraws = nraws
        self.shuffle = shuffle
        self.vocab = vocab
        self.labels = labels
        self.tokenizer = tokenizer
        self.wordNgrams = wordNgrams
        self.buckets = buckets
        self.max_length = max_length
        self.device = device

    def process_oneline(self, line):
        """ process """
        line = line.strip()
        content, label = line.split('\t')
        if content == 0:
            content = "0"
        tokens = self.tokenizer(content.strip())
        seq_len = len(tokens)
        if seq_len > self.max_length:
            tokens = tokens[:self.max_length]

        token_hash_list = [hash_str(token) for token in tokens]
        ngram = addWordNgrams(token_hash_list, self.wordNgrams, self.buckets)
        ngram_pad_size = int((self.wordNgrams - 1) * (self.max_length - self.wordNgrams / 2))

        if len(ngram) > ngram_pad_size:
            ngram = ngram[:ngram_pad_size]
        tokens_to_id = [self.vocab.get(token, self.vocab.get(UNK)) for token in tokens]
        y = self.labels.index(label.strip())

        return tokens_to_id, ngram, y

    def initial(self):
        """init"""
        self.finput = open(self.file_path, 'r', encoding="utf-8")
        self.samples = list()

        for _ in range(self.nraws):
            line = self.finput.readline()
            if line:
                preprocess_data = self.process_oneline(line)
                self.samples.append(preprocess_data)
            else:
                break

        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)
        self.finput.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        one_sample = self.samples[idx]
        return one_sample


class DataGenerator:
    """data generator"""

    def __init__(self, dataset, max_length):
        """init function"""
        self.ids = []
        self.ngrad_ids = []
        self.label = []
        for i in range(len(dataset)):
            ids_item = dataset[i][0]
            ids_item = self.padding(ids_item, max_length)
            ngradids_item = dataset[i][1]
            ngradids_item = self.padding(ngradids_item, max_length)
            self.ids.append(np.array(ids_item))
            self.ngrad_ids.append(np.array(ngradids_item))
            self.label.append(np.array([dataset[i][2]]))

    def __getitem__(self, item):
        return self.ids[item], self.ngrad_ids[item], self.label[item]

    def __len__(self):
        return len(self.ids)

    def padding(self, mylist, maxlen):
        """padding"""
        if len(mylist) > maxlen:
            return mylist[:maxlen]
        return mylist + [0] * (maxlen - len(mylist))


def build_dataloader(dataset, batch_size, max_length, shuffle=False, rank_size=1, rank_id=0, num_parallel_workers=4):
    """build data loader"""
    type_cast_op = C.TypeCast(mstype.int32)
    dataset.initial()
    datagenerator = DataGenerator(dataset, max_length)
    d_iter = ds.GeneratorDataset(datagenerator, ["ids", "ngrad_ids", "label"], shuffle=shuffle, num_shards=rank_size,
                                 shard_id=rank_id, num_parallel_workers=num_parallel_workers)
    d_iter = d_iter.map(operations=type_cast_op, input_columns="ids")
    d_iter = d_iter.map(operations=type_cast_op, input_columns="ngrad_ids")
    d_iter = d_iter.map(operations=type_cast_op, input_columns="label")
    d_iter = d_iter.batch(batch_size, drop_remainder=True)
    return d_iter


def get_time_dif(start_time):
    """get time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
