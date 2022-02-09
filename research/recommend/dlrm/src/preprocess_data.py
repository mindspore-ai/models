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
"""Download raw data and preprocessed data."""
import os
import pickle
import collections
import numpy as np
from mindspore.mindrecord import FileWriter
from model_utils.config import config

class StatsDict():
    """preprocessed data"""

    def __init__(self, field_size, dense_dim, slot_dim, skip_id_convert):
        self.field_size = field_size
        self.dense_dim = dense_dim
        self.slot_dim = slot_dim
        self.skip_id_convert = bool(skip_id_convert)

        self.val_cols = ["val_{}".format(i + 1) for i in range(self.dense_dim)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(self.slot_dim)]

        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}

        self.cat2id_dict = {}

    def get_cat_feature_sizes(self):
        sizes = []
        for _, cat_count_d in self.cat_count_dict.items():
            sizes.append(len(cat_count_d))
        return sizes

    def stats_cats(self, cat_list):
        """Handling cats column

        Get number of every categorical feature.
        """

        assert len(cat_list) == len(self.cat_cols)

        def map_cat_count(i, cat):
            key = self.cat_cols[i]
            self.cat_count_dict[key][cat] += 1

        for i, cat in enumerate(cat_list):
            map_cat_count(i, cat)

    def save_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "wb") as file_wrt:
            pickle.dump(self.cat_count_dict, file_wrt)

    def load_dict(self, dict_path, prefix=""):
        with open(os.path.join(dict_path, "{}cat_count_dict.pkl".format(prefix)), "rb") as file_wrt:
            self.cat_count_dict = pickle.load(file_wrt)
        # print(f"cat_count_dict {self.cat_count_dict}")

    def get_cat2id(self, threshold=100):
        """assign an ID for every categorical feature
        """

        for key, cat_count_d in self.cat_count_dict.items():
            for i, cat_str in enumerate(cat_count_d.keys()):
                self.cat2id_dict[key + "_" + cat_str] = i

        print("cat2id_dict.size:{}".format(len(self.cat2id_dict)))
        print("cat2id.dict.items()[:50]:{}".format(list(self.cat2id_dict.items())[:50]))

    def map_cat2id(self, cats):
        """Cat to id"""
        id_list = []
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
        return id_list


# pylint:enable=missing-docstring
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
            recommendation_dataset_stats_dict.stats_cats(cats)
    recommendation_dataset_stats_dict.save_dict(dict_output_path)

# pylint:enable=missing-docstring
def process_numercal_features(values):
    result = []
    for value in values:
        if value == '':
            result.append(0)
        else:
            v = int(value)
            v = v if v >= 0 else 0
            result.append(v)
    return result

def random_split_trans2mindrecord(input_file_path, output_file_path, recommendation_dataset_stats_dict,
                                  part_rows=2000000, line_per_sample=1000, train_line_count=None,
                                  days=7, dense_dim=13, slot_dim=26):
    """Random split data and save mindrecord

    Args:
        days: Criteo contains 7 days data, approximately mean.
    """
    if train_line_count is None:
        raise ValueError("Please provide training file line count")
    test_size = int(train_line_count // days)
    print(f"test size:{test_size}")
    print("-----------------------" * 10 + "\n" * 2)

    train_data_list = []
    test_data_list = []
    ids_list = []
    wts_list = []
    label_list = []

    writer_train = FileWriter(os.path.join(output_file_path, "train_input_part.mindrecord"), 210) #21
    writer_test = FileWriter(os.path.join(output_file_path, "test_input_part.mindrecord"), 30) #3

    schema = {"label": {"type": "float32", "shape": [-1]}, "feat_vals": {"type": "float32", "shape": [-1]},
              "feat_ids": {"type": "int32", "shape": [-1]}}
    writer_train.add_schema(schema, "CRITEO_TRAIN")
    writer_test.add_schema(schema, "CRITEO_TEST")

    with open(input_file_path, encoding="utf-8") as file_in:
        items_error_size_lineCount = []
        count = 0
        train_part_number = 0
        test_part_number = 0
        for i, line in enumerate(file_in):
            count += 1
            if count % 1000000 == 0:
                print("Have handle {}w lines.".format(count // 10000))
            line = line.strip("\n")
            items = line.split("\t")
            if len(items) != (1 + dense_dim + slot_dim):
                items_error_size_lineCount.append(i)
                continue
            label = float(items[0])
            values = items[1:1 + dense_dim]
            cats = items[1 + dense_dim:]

            assert len(values) == dense_dim, "values.size: {}".format(len(values))
            assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))

            ids, wts = recommendation_dataset_stats_dict.map_cat2id(cats), process_numercal_features(values)

            ids_list.extend(ids)
            wts_list.extend(wts)
            label_list.append(label)

            if count % line_per_sample == 0:
                if i < train_line_count - test_size:
                    train_data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
                                            "feat_vals": np.array(wts_list, dtype=np.float32),
                                            "label": np.array(label_list, dtype=np.float32)
                                            })
                else:
                    test_data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
                                           "feat_vals": np.array(wts_list, dtype=np.float32),
                                           "label": np.array(label_list, dtype=np.float32)
                                           })
                if train_data_list and len(train_data_list) % part_rows == 0:
                    writer_train.write_raw_data(train_data_list)
                    train_data_list.clear()
                    train_part_number += 1

                if test_data_list and len(test_data_list) % part_rows == 0:
                    writer_test.write_raw_data(test_data_list)
                    test_data_list.clear()
                    test_part_number += 1

                ids_list.clear()
                wts_list.clear()
                label_list.clear()

        if train_data_list:
            writer_train.write_raw_data(train_data_list)
        if test_data_list:
            writer_test.write_raw_data(test_data_list)
    writer_train.commit()
    writer_test.commit()

    print("-------------" * 10)
    print("items_error_size_lineCount.size(): {}.".format(len(items_error_size_lineCount)))
    print("-------------" * 10)
    np.save("items_error_size_lineCount.npy", items_error_size_lineCount)


if __name__ == '__main__':

    data_path = config.data_path

    target_field_size = config.dense_dim + config.slot_dim
    stats = StatsDict(field_size=target_field_size, dense_dim=config.dense_dim, slot_dim=config.slot_dim,
                      skip_id_convert=config.skip_id_convert)
    data_file_path = data_path + "train.txt"
    stats_output_path = data_path + "/tats_dict/"
    mkdir_path(stats_output_path)
    statsdata(data_file_path, stats_output_path, stats, dense_dim=config.dense_dim, slot_dim=config.slot_dim)

    stats.load_dict(dict_path=stats_output_path, prefix="")
    stats.get_cat2id(threshold=0)

    config.categorical_feature_sizes = stats.get_cat_feature_sizes()
    print(config.categorical_feature_sizes)

    in_file_path = data_path + "train.txt"
    output_path = data_path + "mindrecord/"
    mkdir_path(output_path)
    random_split_trans2mindrecord(in_file_path, output_path, stats, part_rows=2000000000,
                                  train_line_count=config.train_line_count, line_per_sample=1,
                                  dense_dim=config.dense_dim, slot_dim=config.slot_dim)
