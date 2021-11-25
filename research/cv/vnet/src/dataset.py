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
"""get dataset loader"""
import math
import mindspore
import numpy as np
from src.data_manager import DataManager


class TrainImagelist:
    """train data list"""

    def __init__(self, parameters, data_path, split_file_path):
        self.parameters = parameters
        dataManagerTrain = DataManager(split_file_path,
                                       data_path,
                                       self.parameters)
        dataManagerTrain.loadTrainingData()
        self.train_images = dataManagerTrain.getNumpyImages()
        self.train_labels = dataManagerTrain.getNumpyGT()

    def __getitem__(self, index):
        img = None
        target = None
        keysImg = list(self.train_images.keys())
        keyslabel = list(self.train_labels.keys())
        img = self.train_images[keysImg[index]]
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        target = self.train_labels[keyslabel[index]]
        target = target.astype(np.float32)
        return img, target

    def __len__(self):
        return len(list(self.train_images.keys()))


class InferImagelist:
    """infer data list"""

    def __init__(self, parameters, data_path, split_file_path):
        self.parameters = parameters
        self.dataManagerInfer = DataManager(split_file_path,
                                            data_path,
                                            self.parameters)

        self.dataManagerInfer.loadInferData()
        self.infer_images = self.dataManagerInfer.getNumpyImages()

    def __getitem__(self, index):
        img = None
        keysImg = list(self.infer_images.keys())
        img = self.infer_images[keysImg[index]]
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return mindspore.Tensor(img, mindspore.float32), keysImg[index]

    def __len__(self):
        return len(list(self.infer_images.keys()))


class DistributedSampler:
    """
    Distributed sampler
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset.classes)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank::self.group_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def create_dataset(mode, parameters, data_path, split_file_path, batch_size, num_of_workers=8, num_of_epoch=10,
                   is_distributed=False, rank=0, group_size=1, seed=0):
    """create dataset for train or test"""

    shuffle, drop_last = None, None
    if mode == 'Train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    dataset_generator = TrainImagelist(parameters, data_path, split_file_path)
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset=dataset_generator, rank=rank,
                                     group_size=group_size, shuffle=shuffle, seed=seed)
    dataset = mindspore.dataset.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=shuffle,
                                                 sampler=sampler, num_parallel_workers=num_of_workers)
    dataset = dataset.batch(batch_size, num_parallel_workers=num_of_workers, drop_remainder=drop_last)
    dataset = dataset.repeat(num_of_epoch)
    return dataset
