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
"""Download raw data and preprocessed data."""
import os
import pickle
import collections
import numpy as np


class StatsDict():
    """preprocessed data"""

    def __init__(self, field_size, dense_dim, slot_dim, skip_id_convert):
        self.field_size = field_size
        self.dense_dim = dense_dim
        self.slot_dim = slot_dim
        self.skip_id_convert = bool(skip_id_convert)

        self.val_cols = ["val_{}".format(i + 1) for i in range(self.dense_dim)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(self.slot_dim)]

        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}

        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}

        self.oov_prefix = "OOV"

        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update(
            {self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})

    def stats_vals(self, val_list):
        """Handling weights column"""
        assert len(val_list) == len(self.val_cols)

        def map_max_min(i, val):
            key = self.val_cols[i]
            if val != "":
                if float(val) > self.val_max_dict[key]:
                    self.val_max_dict[key] = float(val)
                if float(val) < self.val_min_dict[key]:
                    self.val_min_dict[key] = float(val)

        for i, val in enumerate(val_list):
            map_max_min(i, val)

    def stats_cats(self, cat_list):
        """Handling cats column"""

        assert len(cat_list) == len(self.cat_cols)

        def map_cat_count(i, cat):
            key = self.cat_cols[i]
            self.cat_count_dict[key][cat] += 1

        for i, cat in enumerate(cat_list):
            map_cat_count(i, cat)

    def save_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.cat_count_dict, file_wrt)

    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}val_max_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_max_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}val_min_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.val_min_dict = pickle.load(file_wrt)
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        print("val_max_dict.items()[:50]:{}".format(list(self.val_max_dict.items())))
        print("val_min_dict.items()[:50]:{}".format(list(self.val_min_dict.items())))

    def get_cat2id(self, threshold=100):
        for key, cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
        print("cat2id_dict.size:{}".format(len(self.cat2id_dict)))
        print("cat2id.dict.items()[:50]:{}".format(list(self.cat2id_dict.items())[:50]))

    def map_cat2id(self, values, cats):
        """Cat to id"""

        def minmax_scale_value(i, val):
            max_v = float(self.val_max_dict["val_{}".format(i + 1)])
            return float(val) * 1.0 / max_v

        id_list = []
        weight_list = []
        for i, val in enumerate(values):
            if val == "":
                id_list.append(i)
                weight_list.append(0)
            else:
                key = "val_{}".format(i + 1)
                id_list.append(self.cat2id_dict[key])
                weight_list.append(minmax_scale_value(i, float(val)))

        for i, cat_str in enumerate(cats):
            key = "cat_{}".format(i + 1) + "_" + cat_str
            if key in self.cat2id_dict:
                if self.skip_id_convert is True:
                    # For the synthetic data, if the generated id is between [0, max_vcoab], but the num examples is l
                    # ess than vocab_size/ slot_nums the id will still be converted to [0, real_vocab], where real_vocab
                    # the actually the vocab size, rather than the max_vocab. So a simple way to alleviate this
                    # problem is skip the id convert, regarding the synthetic data id as the final id.
                    id_list.append(cat_str)
                else:
                    id_list.append(self.cat2id_dict[key])
            else:
                id_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
            weight_list.append(1.0)
        return id_list, weight_list


def mkdir_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def statsdata(file_path, dict_output_path, recommendation_dataset_stats_dict, dense_dim=13, slot_dim=26):
    """Preprocess data and save data"""
    with open(file_path, encoding="utf-8") as file_in:
        errorline_list = []
        count = 0
        for line in file_in:
            count += 1
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != (dense_dim + slot_dim + 1):
                errorline_list.append(count)
                print("Found line length: {}, suppose to be {}, the line is {}".format(len(items),
                                                                                       dense_dim + slot_dim + 1, line))
                continue
            if count % 1000000 == 0:
                print("Have handled {}w lines.".format(count // 10000))
            values = items[1: dense_dim + 1]
            cats = items[dense_dim + 1:]

            assert len(values) == dense_dim, "values.size: {}".format(len(values))
            assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))
            recommendation_dataset_stats_dict.stats_vals(values)
            recommendation_dataset_stats_dict.stats_cats(cats)
    recommendation_dataset_stats_dict.save_dict(dict_output_path)


if __name__ == '__main__':

    data_path = '../data/'
    target_field_size = 39
    stats = StatsDict(field_size=target_field_size, dense_dim=13, slot_dim=26, skip_id_convert=0)
    data_file_path = data_path + "input/train.txt"
    stats_output_path = data_path + "stats_dict/"
    mkdir_path(stats_output_path)

    statsdata(data_file_path, stats_output_path, stats, dense_dim=13, slot_dim=26)
    stats.load_dict(dict_path=stats_output_path, prefix="")
    stats.get_cat2id(threshold=100)
    in_file_path = data_path + "input/train.txt"

    """Get test sets from datasets"""
    train_line_count = 45840617
    if train_line_count is None:
        raise ValueError("Please provide training file line count")
    test_size = 0.1
    test_size = int(train_line_count * test_size)
    all_indices = [i for i in range(train_line_count)]
    np.random.seed(2020)
    np.random.shuffle(all_indices)
    print("all_indices.size:{}".format(len(all_indices)))
    test_indices_set = set(all_indices[:test_size])
    print("test_indices_set.size:{}".format(len(test_indices_set)))
    print("-----------------------" * 10 + "\n" * 2)

    with open(in_file_path, encoding="utf-8") as f:
        fo = open("../data/input/infertest.txt", "w")
        j = 0
        dataline = f.readline()
        while dataline:
            if j in test_indices_set:
                fo.write(dataline)
            j += 1
            dataline = f.readline()
        fo.close()

    fi = open('../data/input/infertest.txt', 'r')
    fo1 = open('../data/input/label.txt', 'w')
    fo2 = open('../data/input/feat_ids.txt', 'w')
    fo3 = open('../data/input/feat_vals.txt', 'w')
    for eachline in fi:
        inferline = eachline.strip("\n")
        item = eachline.split("\t")
        inferlabel = float(item[0])
        infervalue = item[1:1 + 13]
        infercats = item[1 + 13:]
        ids, wts = stats.map_cat2id(infervalue, infercats)
        fo1.write(str(int(inferlabel)) + "\n")
        fo2.write("\t".join(str(id) for id in ids) + "\n")
        fo3.write("\t".join(str(wt) for wt in wts) + "\n")
    fo1.close()
    fo2.close()
    fo3.close()
