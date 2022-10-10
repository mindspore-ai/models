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


import pandas as pd
import mindspore.dataset as ds
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from util import MyDataset


def data_process(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(data[data['rating'] == 3].index)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    train, test = train_test_split(data, test_size=0.2)
    train, valid = train_test_split(train, test_size=0.2)
    return [train, valid, test, data]


def create_dataset(data_set, batch_size=32):

    dataset = ds.GeneratorDataset(data_set, column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset


def get_user_feature(data):
    data_group = data[['user_id', 'rating']].groupby('user_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='user_id')
    return data


def get_item_feature(data):
    data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='movie_id')
    return data


def process_struct_data(data_path):
    data_list = data_process(data_path)
    train, valid, test, data = data_list[0], data_list[1], data_list[2], data_list[3]

    train = get_user_feature(train)
    train = get_item_feature(train)

    valid = get_user_feature(valid)
    valid = get_item_feature(valid)

    test = get_user_feature(test)
    test = get_item_feature(test)

    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
    dense_features = ['user_mean_rating', 'item_mean_rating']

    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        valid[feat] = lbe.transform(valid[feat])
        test[feat] = lbe.transform(test[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    mms.fit(valid[dense_features])
    mms.fit(test[dense_features])
    train[dense_features] = mms.transform(train[dense_features])
    valid[dense_features] = mms.transform(valid[dense_features])
    test[dense_features] = mms.transform(test[dense_features])

    train_dataset_generator = MyDataset(train[sparse_features + dense_features],
                                        train['rating'])
    valid_dataset_generator = MyDataset(valid[sparse_features + dense_features],
                                        valid['rating'])
    test_dataset_generator = MyDataset(test[sparse_features + dense_features],
                                       test['rating'])
    return train_dataset_generator, valid_dataset_generator, test_dataset_generator


def construct_dataset(dataset_generator, batch_size):
    dataset = create_dataset(dataset_generator, batch_size)
    return dataset
