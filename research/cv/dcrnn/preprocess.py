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

import os
import argparse
import numpy as np
import mindspore.dataset as ds

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, transdata):
        return (transdata - self.mean) / self.std
    def get_mean(self):
        return self.mean
    def get_std(self):
        return self.std

def load_dataset(data__path):
    data__set = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data__path, category + '.npz'))
        data__set['x_' + category] = cat_data['x']
        data__set['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data__set['x_train'][..., 0].mean(), std=data__set['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data__set['x_' + category][..., 0] = scaler.transform(data__set['x_' + category][..., 0])
        data__set['y_' + category][..., 0] = scaler.transform(data__set['y_' + category][..., 0])
    data__set['scaler'] = scaler
    return data__set

class GetDatasetGenerator():
    def __init__(self, xs, ys, batch__size, pad_with_last_sample=True, shuffle=False):
        np.random.seed(58)
        self.batch_size = batch__size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch__size - (len(xs) % batch__size)) % batch__size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.__data = xs
        self.__label = ys
    def __getitem__(self, index):
        output = self.__data[index]
        target = self.__label[index]
        return (output, target)
    def __len__(self):
        return len(self.__data)

def get_loader_dataset(datapath):
    data = load_dataset(datapath)
    dataset_generator = GetDatasetGenerator(data['x_val'], data['y_val'], batch__size=64, shuffle=True)
    data_set = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    data_set = data_set.batch(batch_size=64, num_parallel_workers=8)
    return data_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--init-dataset-path', type=str, default=None)
    args = parser.parse_args()
    init_dataset_path = args.init_dataset_path
    print("Export bin files begin!")
    data_path = "./preprocess_Result/data"
    label_path = "./preprocess_Result/label"
    os.makedirs(data_path)
    os.makedirs(label_path)
    dataset = get_loader_dataset(init_dataset_path)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        valdata = item["data"]
        data_file_path = os.path.join(data_path, "data_"+str(i)+".bin")
        valdata.tofile(data_file_path)
        label = item["label"]
        label = label.astype(np.float32)
        label = label.transpose((1, 0, 2, 3))
        label = label[..., :1].reshape(12, 64, 207)
        label_file_path = os.path.join(label_path, "label_"+str(i)+".bin")
        label.tofile(label_file_path)
        i = i + 1
    print("Export bin files finished!")
    