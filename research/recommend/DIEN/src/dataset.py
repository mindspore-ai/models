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
"""
dataset of dien
"""

import os
import random
import gzip
import _pickle as pkl
import numpy as np

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

from .shuffle import shuffle as shuffle_func


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 meta_path,
                 review_path,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None):
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle_func(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        meta_map = {}
        with open(meta_path, "r", encoding='utf-8') as f_meta:
            for line in f_meta:
                arr = line.strip().split("\t")
                if arr[0] not in meta_map:
                    meta_map[arr[0]] = arr[1]
        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx
        self.mid_list_for_random = []
        with open(review_path, "r", encoding='utf-8') as f_review:
            for line in f_review:
                arr = line.strip().split("\t")
                tmp_idx = 0
                if arr[1] in self.source_dicts[1]:
                    tmp_idx = self.source_dicts[1][arr[1]]
                self.mid_list_for_random.append(tmp_idx)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle_func(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def __source_list(self, ss, ss_idx, dict_idx):
        tmp = []
        for fea in ss[ss_idx].split(""):
            m = self.source_dicts[dict_idx][fea] if fea in self.source_dicts[dict_idx] else 0
            tmp.append(m)
        return tmp

    def __next_stop(self):
        """__next_stop"""
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        if not self.source_buffer:
            for _ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = np.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if not self.source_buffer:
            self.end_of_data = False
            self.reset()
            raise StopIteration

    def __next_handle(self, ss, uid, mid, cat, mid_list, cat_list, source, target):
        """
            The function will change the content of source and target.
        """
        noclk_mid_list = []
        noclk_cat_list = []
        for pos_mid in mid_list:
            noclk_tmp_mid = []
            noclk_tmp_cat = []
            noclk_index = 0
            while True:
                noclk_mid_indx = random.randint(
                    0, len(self.mid_list_for_random) - 1)
                noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                if noclk_mid == pos_mid:
                    continue
                noclk_tmp_mid.append(noclk_mid)
                noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                noclk_index += 1
                if noclk_index >= 5:
                    break
            noclk_mid_list.append(noclk_tmp_mid)
            noclk_cat_list.append(noclk_tmp_cat)
        source.append([uid, mid, cat, mid_list, cat_list, noclk_mid_list, noclk_cat_list])
        target.append([float(ss[0]), 1 - float(ss[0])])

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        source = []
        target = []
        self.__next_stop()
        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0

                mid_list = self.__source_list(ss, 4, 1)
                cat_list = self.__source_list(ss, 5, 2)
                # read from source file and map to word index
                if self.minlen and (len(mid_list) <= self.minlen):
                    continue
                if self.skip_empty and (not mid_list):
                    continue

                self.__next_handle(ss, uid, mid, cat, mid_list,
                                   cat_list, source, target)

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
        # all sentence pairs in maxibatch filtered out because of length
        if not source or not target:
            source, target = self.__next__()

        return source, target


def pre_data(p_input, target, maxlen=None):
    """# x: a list of sentences"""
    lengths_x = [len(s[4]) for s in p_input]
    seqs_mid = [inp[3] for inp in p_input]
    seqs_cat = [inp[4] for inp in p_input]
    noclk_seqs_mid = [inp[5] for inp in p_input]
    noclk_seqs_cat = [inp[6] for inp in p_input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, p_input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        x_len = len(lengths_x)
        if x_len < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = np.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = np.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = np.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = np.zeros(
        (n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = np.array([inp[0] for inp in p_input])
    mids = np.array([inp[1] for inp in p_input])
    cats = np.array([inp[2] for inp in p_input])

    sl = np.array(lengths_x)
    target = np.array(target)
    return uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mid_his, noclk_cat_his


def shape0(uids, mids, cats):
    if uids.shape[0] != 128:
        uids = np.pad(uids, (0, 128 - uids.shape[0]), constant_values=(0, 0))
    if mids.shape[0] != 128:
        mids = np.pad(mids, (0, 128 - mids.shape[0]), constant_values=(0, 0))
    if cats.shape[0] != 128:
        cats = np.pad(cats, (0, 128 - cats.shape[0]), constant_values=(0, 0))
    return uids, mids, cats


def prepare_data(p_input, target, maxlen=None, return_neg=False, mode=""):
    TRAIN_MODE = "train"
    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mid_his, noclk_cat_his = pre_data(
        p_input, target, maxlen)
    if mode == TRAIN_MODE:
        uids, mids, cats = shape0(uids, mids, cats)

        if mid_his.shape[0] != 128 or mid_his.shape[1] != 100:
            mid_his = np.pad(mid_his, ((0, 128 - mid_his.shape[0]), (0, 100 - mid_his.shape[1])),
                             constant_values=(0, 0))
        if cat_his.shape[0] != 128 or cat_his.shape[1] != 100:
            cat_his = np.pad(cat_his, ((0, 128 - cat_his.shape[0]), (0, 100 - cat_his.shape[1])),
                             constant_values=(0, 0))
        if mid_mask.shape[0] != 128 or mid_mask.shape[1] != 100:
            mid_mask = np.pad(mid_mask, ((0, 128 - mid_mask.shape[0]), (0, 100 - mid_mask.shape[1])),
                              constant_values=(0, 0))
        if target.shape[0] != 128 or target.shape[1] != 2:
            target = np.pad(target, ((0, 128 - target.shape[0]), (0, 2 - target.shape[1])), constant_values=(0, 0))
        if sl.shape[0] != 128:
            sl = np.pad(sl, (0, 128 - sl.shape[0]), constant_values=(0, 0))
        if noclk_mid_his.shape[0] != 128 or noclk_mid_his.shape[1] != 100 or noclk_mid_his.shape[2] != 5:
            noclk_mid_his = np.pad(noclk_mid_his,
                                   ((0, 128 - noclk_mid_his.shape[0]), (0, 100 - noclk_mid_his.shape[1]),
                                    (0, 5 - noclk_mid_his.shape[2])),
                                   constant_values=(0, 0))
        if noclk_cat_his.shape[0] != 128 or noclk_cat_his.shape[1] != 100 or noclk_cat_his.shape[2] != 5:
            noclk_cat_his = np.pad(noclk_cat_his,
                                   ((0, 128 - noclk_cat_his.shape[0]), (0, 100 - noclk_cat_his.shape[1]),
                                    (0, 5 - noclk_cat_his.shape[2])),
                                   constant_values=(0, 0))
        if return_neg:
            return uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mid_his, noclk_cat_his
        return uids, mids, cats, mid_his, cat_his, mid_mask, target, sl
    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mid_his, noclk_cat_his
    return uids, mids, cats, mid_his, cat_his, mid_mask, target, sl


def create_mindrecord(train_data, maxlen, MINDRECORD_FILE, mode=""):
    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
        os.remove(MINDRECORD_FILE + ".db")

    writer = FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
    nlp_schema = {"mid_mask": {"type": "float32", "shape": [128, 100]},
                  "uids": {"type": "int32", "shape": [128]},
                  "mids": {"type": "int32", "shape": [128]},
                  "cats": {"type": "int32", "shape": [128]},
                  "mid_his": {"type": "int32", "shape": [128, 100]},
                  "cat_his": {"type": "int32", "shape": [128, 100]},
                  "sl": {"type": "int32", "shape": [128]},
                  "noclk_mids": {"type": "int32", "shape": [128, 100, 5]},
                  "noclk_cats": {"type": "int32", "shape": [128, 100, 5]},
                  "target": {"type": "float32", "shape": [128, 2]}}
    writer.add_schema(nlp_schema, "it is a preprocessed nlp dataset")

    data = []
    for src, tgt in train_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt,
                                                                                                        maxlen,
                                                                                                        return_neg=True,
                                                                                                        mode=mode)
        uids = uids.astype(np.int32)
        mids = mids.astype(np.int32)
        cats = cats.astype(np.int32)
        mid_his = mid_his.astype(np.int32)
        cat_his = cat_his.astype(np.int32)
        sl = sl.astype(np.int32)
        mid_mask = mid_mask.astype(np.float32)
        target = target.astype(np.float32)
        noclk_mids = noclk_mids.astype(np.int32)
        noclk_cats = noclk_cats.astype(np.int32)
        sample = {"mid_mask": mid_mask,
                  "uids": uids,
                  "mids": mids,
                  "cats": cats,
                  "mid_his": mid_his,
                  "cat_his": cat_his,
                  "sl": sl,
                  "noclk_mids": noclk_mids,
                  "noclk_cats": noclk_cats,
                  "target": target}
        data.append(sample)
    writer.write_raw_data(data)
    writer.commit()


def create_dataset(MINDRECORD_FILE):
    data_set = ds.MindDataset(MINDRECORD_FILE)
    return data_set
