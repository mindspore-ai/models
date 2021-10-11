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
"""mnist dataset"""
import os
import struct
import numpy as np
from mindspore import dataset as ds


def decode_idx3_ubyte(idx3_ubyte_file):
    """parse .idx3_ubyte file"""
    # get the binary date
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # parse date set
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i1 in range(num_images):
        images[i1] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """parse .idx1_ubyte"""
    # get the binary date
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    _, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # parse date set
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i2 in range(num_images):
        labels[i2] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_data(fp):
    images = decode_idx3_ubyte(fp)
    return images


def load_label(fp):
    labels = decode_idx1_ubyte(fp)
    return labels


class DatasetGenerator:
    """mnist dataset"""
    def __init__(self, data_path, usage):
        if usage == 'train':
            self.data = load_data(os.path.join(data_path, 'train-images-idx3-ubyte')).astype("uint8")
            self.label = load_label(os.path.join(data_path, 'train-labels-idx1-ubyte')).astype("uint8")
        elif usage == 'test':
            self.data = load_data(os.path.join(data_path, 't10k-images-idx3-ubyte')).astype("uint8")
            self.label = load_label(os.path.join(data_path, 't10k-labels-idx1-ubyte')).astype("uint8")
        else:
            self.data = np.concatenate((load_data(os.path.join(data_path, 'train-images-idx3-ubyte')).astype("uint8"),
                                        load_data(os.path.join(data_path, 't10k-images-idx3-ubyte')).astype("uint8")),
                                       axis=0)
            self.label = np.concatenate((load_label(os.path.join(data_path, 'train-labels-idx1-ubyte')).astype("uint8"),
                                         load_label(os.path.join(data_path, 't10k-labels-idx1-ubyte')).astype("uint8")),
                                        axis=0)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


def create_dataset(data_path, usage, batch_size=1):
    """create dataset"""
    dataset_generator = DatasetGenerator(data_path, usage)

    mnist_ds = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=False)

    mnist_ds = mnist_ds.batch(batch_size, True)
    mnist_ds = mnist_ds.repeat(1)
    return mnist_ds
