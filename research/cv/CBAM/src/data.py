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

"""Data preprocessing"""

import os
import pickle
import numpy as np

from mindspore.communication.management import get_rank, get_group_size
import mindspore.dataset as de
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms as C


def _get_rank_info(run_distribute):
    """get rank size and rank id"""
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if run_distribute:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0
    return rank_size, rank_id


class Get_Data():
    """
    The data is preprocessed before being converted to midnspore.
    """
    def __init__(self, data_path, snr=None, training=True):
        self.data_path = data_path + 'RML2016.10a_dict.pkl'
        self.snr = snr
        self.do_train = training
        self.data_file = open(self.data_path, 'rb')
        self.all_data = pickle.load(self.data_file, encoding='iso-8859-1')
        self.snrs, self.mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.all_data.keys())))), [1, 0])

    def to_one_hot(self, label_index):
        """generate one hot label"""
        one_hot_label = np.zeros([len(label_index), max(label_index) + 1])
        one_hot_label[np.arange(len(label_index)), label_index] = 1
        return one_hot_label

    def get_loader(self):
        train_data = []
        train_label = []

        test_data = []
        test_label = []

        train_index = []

        for key in self.all_data.keys():
            v = self.all_data[key]
            train_data.append(v[:v.shape[0]//2])
            test_data.append(v[v.shape[0]//2:])
            for i in range(self.all_data[key].shape[0]//2):
                train_label.append(key)
                test_label.append(key)

        train_data = np.vstack(train_data)
        test_data = dict(zip(self.all_data.keys(), test_data))

        for i in range(0, 110000):
            train_index.append(i)

        ds_label = []
        if self.do_train:
            ds = train_data
            ds = np.expand_dims(ds, axis=1)
            ds_label = self.to_one_hot(list(map(lambda x: self.mods.index(train_label[x][0]), train_index)))
        else:
            ds = []
            for mod in self.mods:
                ds.append((test_data[(mod, self.snr)]))
                ds_label += [mod] * test_data[(mod, self.snr)].shape[0]

            ds = np.vstack(ds)
            ds = np.expand_dims(ds, axis=1)
            ds_label = self.to_one_hot(list(self.mods.index(x) for x in ds_label))

        return ds, ds_label


def create_dataset(data_path,
                   batch_size=1,
                   training=True,
                   snr=None,
                   target="Ascend",
                   run_distribute=False):
    """create dataset for train or eval"""
    if target == "Ascend":
        device_num, rank_id = _get_rank_info(run_distribute)
    if training:
        getter = Get_Data(data_path=data_path, snr=None, training=training)
        data = getter.get_loader()
    else:
        getter = Get_Data(data_path=data_path, snr=snr, training=training)
        data = getter.get_loader()

    dataset_column_names = ["data", "label"]
    if target != "Ascend" or device_num == 1:
        if training:
            ds = de.NumpySlicesDataset(data=data,
                                       column_names=dataset_column_names,
                                       shuffle=True)
        else:
            ds = de.NumpySlicesDataset(data=data,
                                       column_names=dataset_column_names,
                                       shuffle=False)

    else:
        if training:
            ds = de.NumpySlicesDataset(data=data,
                                       column_names=dataset_column_names,
                                       shuffle=True,
                                       num_shards=device_num,
                                       shard_id=rank_id)
        else:
            ds = de.NumpySlicesDataset(data=data,
                                       column_names=dataset_column_names,
                                       shuffle=False,
                                       num_shards=device_num,
                                       shard_id=rank_id)
    ds_label = [
        C.TypeCast(mstype.float32)
    ]
    ds = ds.map(operations=ds_label, input_columns=["label"])
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds


if __name__ == '__main__':
    data_getter = Get_Data(data_path='../data/', snr=2, training=False)
    ms_ds = data_getter.get_loader()
