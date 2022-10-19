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
'''Data set preprocessing'''
import struct
import numpy as np
from src.param_parse import parameter_parser
from mindspore import dataset as ds
from mindspore.communication.management import get_rank, get_group_size

opt = parameter_parser()

# train set
train_images_idx3_ubyte_file = opt.data_path + 'train-images.idx3-ubyte'
# train labels set
train_labels_idx1_ubyte_file = opt.data_path + 'train-labels.idx1-ubyte'

# test set
test_images_idx3_ubyte_file = opt.data_path + 't10k-images.idx3-ubyte'
# test labels set
test_labels_idx1_ubyte_file = opt.data_path + 't10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """parse .idx3_ubyte file"""
    # get the binary date
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic numbers :%d, image numbers: %d, image size: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # parse date set
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    for i1 in range(num_images):
        if (i1 + 1) % 10000 == 0:
            print('parsed %d' % (i1 + 1) + 'images')
            print(offset)
        images[i1] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """parse .idx1_ubyte"""
    # get the binary date
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic numbers :%d, image numbers : %d' % (magic_number, num_images))

    # parse date set
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i2 in range(num_images):
        labels[i2] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_data():
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    return train_images


def load_train_label():
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    return train_labels

def load_valid_data():
    valid_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    return valid_images[-1000:]

def load_valid_label():
    valid_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    return valid_labels[-1000:]

def load_test_data():
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    return test_images

class DatasetGenerator:
    def __init__(self):
        self.data = load_train_data().astype("float32")
        self.label = load_train_label().astype("float32")

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class DatasetGenerator_valid:
    def __init__(self):
        self.data = load_valid_data().astype("float32")
        self.label = load_valid_label().astype("float32")

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


def create_dataset_train(batch_size=5, repeat_size=1, latent_size=100):
    """create dataset train"""
    dataset_generator = DatasetGenerator()
    dataset1 = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True)
    mnist_ds = dataset1.map(
        operations=lambda x: (
            x.astype("float32"),
            np.random.normal(size=(latent_size)).astype("float32")
        ),
        output_columns=["image", "latent_code"]
    )
    mnist_ds = mnist_ds.project(["image", "latent_code"])
    mnist_ds = mnist_ds.batch(batch_size, True)
    mnist_ds = mnist_ds.repeat(1)
    return mnist_ds

def create_dataset_train_dis(batch_size=5, repeat_size=1, latent_size=100):
    """create dataset train"""
    dataset_generator = DatasetGenerator()
    dataset1 = ds.GeneratorDataset(dataset_generator, ["image", "label"],
                                   shuffle=True, num_shards=get_group_size(), shard_id=get_rank())
    mnist_ds = dataset1.map(
        operations=lambda x: (
            x.astype("float32"),
            np.random.normal(size=(latent_size)).astype("float32")
        ),
        output_columns=["image", "latent_code"]
    )
    mnist_ds = mnist_ds.project(["image", "latent_code"])
    mnist_ds = mnist_ds.batch(batch_size, True)
    mnist_ds = mnist_ds.repeat(1)
    return mnist_ds


def create_dataset_valid(batch_size=5, repeat_size=1, latent_size=100):
    """create dataset valid"""
    dataset_generator = DatasetGenerator_valid()
    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=False)
    mnist_ds = dataset.map(
        operations=lambda x: (
            x[-10000:].astype("float32"),
            np.random.normal(size=(latent_size)).astype("float32")
        ),
        output_columns=["image", "latent_code"]
    )
    mnist_ds = mnist_ds.project(["image", "latent_code"])
    mnist_ds = mnist_ds.batch(batch_size, True)
    mnist_ds = mnist_ds.repeat(1)
    return mnist_ds
