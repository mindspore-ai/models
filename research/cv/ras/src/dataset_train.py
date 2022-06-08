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
from mindspore.communication import get_rank, get_group_size
from PIL import Image

image_size = (352, 352)


class IterDatasetGenerator():
    """
    Specific train data processing class
    """
    def __init__(self, image_path, label_path):
        self.__data_name = get_data_dirlist(image_path)
        self.__label_name = get_data_dirlist(label_path)
        self.image_number = len(self.__data_name)
        self.__index = 0

    def __getitem__(self, index):
        image = get_data(self.__data_name[index])
        label = get_objs(self.__label_name[index])
        image, label = self.cv_random_flip(image, label)
        image, label = self.cv_random_rotate(image, label)
        image = self.handle_image(image)
        label = self.handle_label(label)

        return image, label

    def __next__(self):
        if self.__index >= len(self.__data_name):
            raise StopIteration
        item = (self.__data_name[self.__index], self.__label_name[self.__index])
        self.__index += 1
        return item

    def handle_image(self, img):
        """

        Args:
            img:

        Returns: a array

        """
        img = np.array(img)
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

        return img

    def handle_label(self, label):
        """

        Args:
            label:

        Returns:
            a array, shape = (3, image_size[0], image_size[1])

        """
        label = np.array(label)
        label = label.astype(np.float32)
        resize = C.Resize(image_size)
        label = resize(label)
        label = label.reshape(-1, image_size[0], image_size[1]) / 255.0

        return label

    def cv_random_flip(self, image, label):
        if np.random.randint(2) == 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return image, label

    def cv_random_rotate(self, image, label):
        rorate_degree = np.random.random() * 2 * 10 - 10
        image = image.rotate(rorate_degree, Image.BILINEAR)
        label = label.rotate(rorate_degree, Image.NEAREST)
        return image, label

    def __len__(self):
        return len(self.__data_name)


class TrainDataLoader():
    """
    data loader, and shuffle, batch
    """

    def __init__(self, image_path, label_path, batch_size, df):
        self.batch_size = batch_size
        self.image_path = image_path
        self.label_path = label_path
        self.dataset_generator = IterDatasetGenerator(image_path, label_path)
        self.data_number = self.dataset_generator.image_number
        self.num_parallel_workers = 4

        if df == "YES":
            rank_id = get_rank()
            rank_size = get_group_size()
            self.dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], num_shards=rank_size,
                                               shard_id=rank_id, shuffle=True)
        else:
            self.dataset = ds.GeneratorDataset(self.dataset_generator, ["data", "label"], shuffle=True,
                                               num_parallel_workers=self.num_parallel_workers)

        self.dataset = self.dataset.batch(batch_size=self.batch_size, drop_remainder=True)


def get_data(filename):
    with open(filename, 'rb') as f:
        img = Image.open(f)
        img = img.convert("RGB")
    return img


def get_objs(filename):
    with open(filename, 'rb') as f:
        objs = Image.open(f)
        objs = objs.convert("L")
    return objs


def get_data_dirlist(data_path):
    data_dirlist = []
    for filename in os.listdir(data_path):
        file_path = data_path + filename
        data_dirlist.append(file_path)
    data_dirlist_sorted = sorted(data_dirlist)
    return data_dirlist_sorted
