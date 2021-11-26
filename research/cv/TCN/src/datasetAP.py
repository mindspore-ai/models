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
"""
######################## read adding problem dataset ########################
"""
import os
import numpy as np
import mindspore.dataset as ds



def read_dataset(datapath, N, seq_length, mode='train'):
    """load adding problem dataset"""
    X_path = os.path.join(datapath, 'traindata.bin')
    Y_path = os.path.join(datapath, 'trainlabel.bin')
    if mode == 'test':
        X_path = os.path.join(datapath, 'testdata.bin')
        Y_path = os.path.join(datapath, 'testlabel.bin')

    X = np.fromfile(X_path, dtype=np.float32)
    Y = np.fromfile(Y_path, dtype=np.float32)
    X.shape = N, 2, seq_length
    Y.shape = N, 1

    return X, Y


class DatasetGenerator:
    """generator dataset"""

    def __init__(self, datapath, N, seq, mode='train'):
        self.data, self.label = read_dataset(datapath, N, seq, mode)

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return len(self.data)


def create_datasetAP(datapath, N, seq, mode, batch):
    """return dataset"""
    dataset_generator = DatasetGenerator(datapath, N, seq, mode)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
    dataset = dataset.batch(batch)

    return dataset
