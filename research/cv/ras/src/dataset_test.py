"""
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
"""

import os
import numpy as np
import mindspore.dataset.vision as C
import mindspore.dataset as ds
from PIL import Image

image_size = (352, 352)


class IterDatasetGenerator():
    """


    Specific test data processing class

    """
    def __init__(self, image_path):
        self.__data, self.__data_org = dataloader(image_path)
        self.image_number = len(self.__data)
        self.__index = 0

    def __getitem__(self, index):
        return self.__data[index], self.__data_org[index]

    def __next__(self):
        if self.__index >= len(self.__data):
            raise StopIteration
        item = (self.__data[self.__index], self.__data_org[self.__index])
        self.__index += 1
        return item

    def __len__(self):
        return len(self.__data)


class TrainDataLoader():
    """
    data loader, and shuffle, batch
    """
    def __init__(self, image_path, batch_size=1):
        self.batch_size = batch_size
        self.image_path = image_path
        self.dataset_generator = IterDatasetGenerator(image_path)
        self.data_number = self.dataset_generator.image_number
        self.dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "data_org"], shuffle=False)
        self.dataset = self.dataset.batch(batch_size=self.batch_size)


def get_data(filename):
    """

    Args:
        filename: a path of data

    Returns:
        a array, shape = (3,image_height,image_width)

    """
    with open(filename, 'rb') as f:
        img = Image.open(f)
        img = img.convert("RGB")
    img = np.array(img)
    img_org = img
    img = img.astype(np.float32)
    img = img / 255.0
    resize = C.Resize(image_size)
    img = resize(img)
    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]

    normal = C.Normalize(mean=mean_, std=std_)
    img = normal(img)
    hwc2chw = C.HWC2CHW()
    img = hwc2chw(normal(img)).astype(np.float32)

    return img, img_org


def get_data_dirlist(data_path):
    data_dirlist = []
    for filename in os.listdir(data_path):
        file_path = data_path + filename
        data_dirlist.append(file_path)
    data_dirlist_sorted = sorted(data_dirlist)
    return data_dirlist_sorted


def dataloader(data_path):
    data = []
    data_orgs = []
    data_dirlist_sorted = get_data_dirlist(data_path)
    for file_path in data_dirlist_sorted:
        data_, data_org = get_data(file_path)
        data.append(data_)
        data_orgs.append(data_org)
    return data, data_orgs
