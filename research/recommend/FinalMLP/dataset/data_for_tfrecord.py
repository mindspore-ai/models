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

import random
import os
import stat
import tensorflow as tf


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_data(data_file, w_data_f):
    data_counter = 1
    file_counter = 0
    with os.fdopen(os.open(data_file, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "r") as valid_f:
        out_buf = []
        for line in valid_f:

            out_buf.append(line.strip())
            data_counter += 1
            if data_counter % 100000 == 0:
                with os.fdopen(os.open(f"{w_data_f}/part-r-{file_counter}.csv",
                                       os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR),
                               "w") as fout:
                    fout.write("\n".join(out_buf))
                out_buf = []
                file_counter += 1
        if out_buf:
            with os.fdopen(os.open(f"{w_data_f}/part-r-{file_counter}.csv",
                                   os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR),
                           'w') as fout:
                fout.write("\n".join(out_buf))


def make_feature_map(train_path, feature_map_path, sep=',', header=True):
    if isinstance(train_path, list):
        train_file_list = train_path
    elif os.path.isdir(train_path):
        train_file_list = [os.path.join(train_path, file_name) for file_name in os.listdir(train_path)]
    else:
        train_file_list = [train_path]
    feature_count_dict = dict()
    for train_id_file in train_file_list:
        with os.fdopen(os.open(train_id_file, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "r") as rf:
            for idx, line in enumerate(rf):
                if header and idx == 0:
                    continue
                fields_value = line.strip().split(sep)
                fields_key_values = dict(zip(fields, fields_value))

                for field_name, value in fields_key_values.items():
                    if field_name not in song_dense_features:
                        feature_name = field_name + "," + value
                        if feature_name not in feature_count_dict:
                            feature_count_dict[feature_name] = 1
                        else:
                            feature_count_dict[feature_name] += 1
    sorted_feature_count_dict = dict(sorted(feature_count_dict.items(), key=lambda x: x[1], reverse=True))
    with os.fdopen(os.open(feature_map_path, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "w") as wf:
        for fi, sparse_field in enumerate(song_sparse_feature):
            w_line = sparse_field + "," + "def_val" + "\t" + str(fi) + "\n"
            wf.write(w_line)
        for i, feature_name in enumerate(sorted_feature_count_dict.keys()):
            w_line = feature_name + "\t" + str(i + len(song_sparse_feature)) + "\n"
            wf.write(w_line)


def get_feature_map(feature_map_path):
    feature_map_dict = dict()
    with os.fdopen(os.open(feature_map_path, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "r") as rf:
        for r_line in rf:
            feature_name_value = r_line.strip().split("\t")
            feature_map_dict[feature_name_value[0]] = int(feature_name_value[1])
    return feature_map_dict


def list_file(path):
    files = []
    for file in os.listdir(path):
        if file.startswith('_') or file.startswith('.'):
            continue
        files.append(os.path.join(path, file))
    if files:
        raise ValueError("input file is empty, please check if local file is empty or not")
    return files


def reverse_tfrecord(data_path, out_data_file, row_number, feature_map_dict, sep=',', header=True):
    if isinstance(data_path, list):
        file_list = data_path
    elif os.path.isdir(data_path):
        file_list = list_file(data_path)
    else:
        file_list = [data_path]
    for i, _ in enumerate(file_list):
        file = file_list[i]
        options = tf.io.TFRecordOptions(compression_type=None)
        writer = tf.io.TFRecordWriter(f"{out_data_file}/part-r-{i}.tfrecord", options)
        with os.fdopen(os.open(file, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), "r") as rf:
            for idx, line in enumerate(rf):
                if header and idx == 0:
                    continue
                feature_dict = dict()
                fields_value = line.strip().split(sep)
                fields_key_values = dict(zip(fields, fields_value))
                row_number += 1
                sparse_feature_list = []
                dense_feature_list = []
                for field, value in fields_key_values.items():
                    if field in song_dense_features:
                        dense_feature_list.append(float(value))
                    elif field == "label":
                        continue
                    else:
                        sparse_feature_list.append(
                            feature_map_dict.get(field + "," + value, feature_map_dict[field + ",def_val"]))

                feature_dict['feat_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_feature_list))
                feature_dict['dense_fea'] = tf.train.Feature(float_list=tf.train.FloatList(value=dense_feature_list))
                feature_dict['feat_vals'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[1.0 for i in sparse_feature_list]))
                feature_dict['label'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(fields_key_values["label"])]))
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
        writer.close()
    return row_number

if __name__ == '__main__':
    random.seed(2022)

    fields = "label,user_id,item_id,tag_id".split(',')
    song_dense_features = []
    song_sparse_feature = set(fields) - set(song_dense_features)
    song_sparse_feature.remove("label")

    VALID_FILE_CSV = "dataset/Movielenslatest_x1/data/valid.csv"
    TRAIN_FILE_CSV = "dataset/Movielenslatest_x1/data/train.csv"
    TEST_FILE_CSV = "dataset/Movielenslatest_x1/data/test.csv"

    VALID_FILE_TF = r"dataset/Movielenslatest_x1\tfrecord_mind\valid"
    TRAIN_FILE_TF = r"dataset/Movielenslatest_x1\tfrecord_mind\train"
    TEST_FILE_TF = r"dataset/Movielenslatest_x1\tfrecord_mind\test"
    FEATURE_MAP_PT = r"dataset/Movielenslatest_x1\tfrecord_mind\feature_map.txt"

    check_path(VALID_FILE_TF)
    check_path(TRAIN_FILE_TF)
    check_path(TEST_FILE_TF)

    make_feature_map([TRAIN_FILE_CSV, VALID_FILE_CSV, TEST_FILE_CSV], FEATURE_MAP_PT, sep=',', header=True)
    feature_map_dt = get_feature_map(FEATURE_MAP_PT)

    ROW_NUM = reverse_tfrecord(VALID_FILE_CSV, VALID_FILE_TF, 1, feature_map_dt, sep=',', header=True)
    ROW_NUM = reverse_tfrecord(TEST_FILE_CSV, TEST_FILE_TF, ROW_NUM, feature_map_dt, sep=',', header=True)
    ROW_NUM = reverse_tfrecord(TRAIN_FILE_CSV, TRAIN_FILE_TF, ROW_NUM, feature_map_dt, sep=',', header=True)
