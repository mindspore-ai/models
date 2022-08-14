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
Data operations, will be used in train.py and eval.py
"""
import os
import h5py
import numpy as np
import scipy.io as sio
import mindspore.dataset as ds
from mindspore import dtype as mstype
import mindspore.dataset.transforms.c_transforms as C2


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def h5_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
    return data, label


def open_list_file(list_dir):
    samples_txt = open(list_dir).readlines()
    return samples_txt


def write_samples_division(data_list, data_loc, samples_txt):
    with open(data_list, 'a') as f:
        for loc in data_loc:
            f.write(samples_txt[loc])


def max_min_normalization(data):
    max_num = data.max()
    min_num = data.min()
    nl_data = (data-min_num) / (max_num-min_num)

    return nl_data


def padding(data, window_size):
    [m, n, c] = data.shape
    start_id = int(np.floor(window_size/2))
    pad_data = np.zeros([m+window_size-1, n+window_size-1, c])
    pad_data[start_id: start_id+m, start_id: start_id+n] = data

    return pad_data


def samples_extraction(source_dir, save_dir, data_list_dir, window_size):
    """
    h5 to npy file
    """
    hsi_data, hsi_gt = h5_loader(source_dir)
    hsi_data = max_min_normalization(hsi_data)
    s = window_size
    hsi_data = padding(hsi_data, s)

    [m, n] = hsi_gt.shape

    make_if_not_exist(save_dir)
    delete_if_exist(data_list_dir)

    for i in range(m):
        for j in range(n):
            if hsi_gt[i, j] > 0:
                label = hsi_gt[i, j]
                data = hsi_data[i:i + s, j:j + s, :].transpose([2, 0, 1])[np.newaxis]
                save_name = save_dir + 'samples_{}_{}.npy'.format(i + 1, j + 1)
                np.save(save_name, data)
                with open(data_list_dir, 'a') as f:
                    f.write(save_name + ' {}\n'.format(label))


def samples_division(list_dir, train_split_dir):
    """
    cross validation division
    """
    samples_txt = open_list_file(list_dir)
    train_txt = open(train_split_dir).readlines()

    label_array = np.array([f.split(' ')[-1][:-1] for f in samples_txt], int)
    train_list = list_dir[: -4] + '_train.txt'
    test_list = list_dir[: -4] + '_test.txt'
    test_list_part = list_dir[: -4] + '_test_part.txt'
    delete_if_exist(train_list)
    delete_if_exist(test_list)
    delete_if_exist(test_list_part)

    for i in range(1, label_array.max()+1):
        class_i_coord = np.where(label_array == i)
        samples_num_i = class_i_coord[0].size
        train_num_i = int(train_txt[i].split()[-1])
        kk = np.random.permutation(samples_num_i)
        if train_num_i < samples_num_i:
            train_loc = class_i_coord[0][kk[:train_num_i]]
            test_loc = class_i_coord[0][kk[train_num_i:]]
            test_part_loc = class_i_coord[0][kk[train_num_i:train_num_i*2]]
        else:
            train_num_i = samples_num_i/2 + 1
            train_loc = class_i_coord[0][kk[:train_num_i]]
            test_loc = class_i_coord[0][kk[train_num_i:]]
            test_part_loc = class_i_coord[0][kk[train_num_i:train_num_i*2]]

        write_samples_division(train_list, train_loc, samples_txt)
        write_samples_division(test_list, test_loc, samples_txt)
        write_samples_division(test_list_part, test_part_loc, samples_txt)


def samples_division_cv(list_dir, train_split_dir, val_split_dir):
    """
    division
    """
    samples_txt = open(list_dir).readlines()
    train_txt = open(train_split_dir).readlines()
    val_txt = open(val_split_dir).readlines()

    label_array = np.array([f.split(' ')[-1][:-1] for f in samples_txt], int)
    train_list = list_dir[: -4] + '_train.txt'
    val_list = list_dir[:-4]+'_val.txt'
    test_list = list_dir[: -4] + '_test.txt'
    delete_if_exist(train_list)
    delete_if_exist(test_list)
    delete_if_exist(val_list)
    for i in range(1, label_array.max()+1):
        class_i_coord = np.where(label_array == i)
        samples_num_i = class_i_coord[0].size
        train_num_i = int(train_txt[i].split()[-1])
        val_num_i = int(val_txt[i].split()[-1])
        kk = np.random.permutation(samples_num_i)
        train_loc = class_i_coord[0][kk[:train_num_i-val_num_i]]
        val_loc = class_i_coord[0][kk[train_num_i-val_num_i: train_num_i]]
        test_loc = class_i_coord[0][kk[train_num_i:]]

        write_samples_division(train_list, train_loc, samples_txt)
        write_samples_division(test_list, test_loc, samples_txt)
        write_samples_division(val_list, val_loc, samples_txt)


def mat2h5(config):
    """
    mat to h5 file
    """
    dataset_name = config.dataset_name
    base_path = config.data_path
    mat_dir = os.path.join(base_path, 'data_mat')
    h5_dir = os.path.join(base_path, 'data_h5')
    if dataset_name == 'Salinas':
        dataset_mat_dir = os.path.join(mat_dir, '{name}/{name}_corrected.mat'.format(name=dataset_name))
        dataset_gt_dir = os.path.join(mat_dir, '{name}/{name}_gt.mat'.format(name=dataset_name))
        dataset_h5_save_dir = os.path.join(h5_dir, '{}.h5'.format(dataset_name))
    elif dataset_name == 'Indian':
        dataset_mat_dir = os.path.join(mat_dir, '{name}/{name}_pines_corrected.mat'.format(name=dataset_name))
        dataset_gt_dir = os.path.join(mat_dir, '{name}/{name}_pines_gt.mat'.format(name=dataset_name))
        dataset_h5_save_dir = os.path.join(h5_dir, '{}.h5'.format(dataset_name))
    elif dataset_name == 'WHU_Hi_HongHu':
        dataset_mat_dir = os.path.join(mat_dir, '{name}/{name}.mat'.format(name=dataset_name))
        dataset_gt_dir = os.path.join(mat_dir, '{name}/{name}_gt.mat'.format(name=dataset_name))
        dataset_h5_save_dir = os.path.join(h5_dir, '{}.h5'.format(dataset_name))
    hsi_data = sio.loadmat(dataset_mat_dir)[config.dataset_HSI]
    hsi_gt = sio.loadmat(dataset_gt_dir)[config.dataset_gt]
    with h5py.File(dataset_h5_save_dir, 'w') as f:
        f['data'] = hsi_data
        f['label'] = hsi_gt


def h52npy(config):
    """
    h5 to npy and divide
    """
    dataset_name = config.dataset_name
    base_path = config.data_path
    samples_dir = os.path.join(base_path, 'samples')
    source_dir = os.path.join(base_path, 'data_h5')
    dataset_source_dir = os.path.join(source_dir, '{}.h5'.format(dataset_name))
    samples_save_dir = samples_dir + '/{}/'.format(dataset_name)
    data_list_dir = './data_list/{}.txt'.format(dataset_name)
    window_size = config.window_size
    train_split_dir = './data_list/{}_split.txt'.format(dataset_name)
    val_split_dir = './data_list/{}_split_val.txt'.format(dataset_name)

    samples_extraction(dataset_source_dir, samples_save_dir, data_list_dir, window_size)
    # samples_division(data_list_dir, train_split_dir)
    samples_division_cv(data_list_dir, train_split_dir, val_split_dir)


class DatasetGenerator:
    """
    preprocess dataset
    """
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

    def __getitem__(self, index):
        sample_path = self.list_txt[index].split(' ')
        data_path = sample_path[0]
        label = sample_path[1][:-1]

        data = np.load(data_path)

        label = int(label) - 1
        return data, label

    def __len__(self):
        return self.length


def create_dataset(config, data_path, do_train=True):
    """
    create dataset
    """
    mat2h5(config)
    h52npy(config)
    batch_size = config.batch_size
    type_cast_label = C2.TypeCast(mstype.int32)
    type_cast_data = C2.TypeCast(mstype.float32)
    data_generator = DatasetGenerator(data_path)
    dataset = ds.GeneratorDataset(
        source=data_generator, column_names=["data", "label"], num_parallel_workers=8, shuffle=do_train)
    dataset = dataset.map(operations=type_cast_label, input_columns="label", num_parallel_workers=8)
    dataset = dataset.map(operations=type_cast_data, input_columns="data", num_parallel_workers=8)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=do_train)
    return dataset
