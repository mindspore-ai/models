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
""" preprocess """

import os
import argparse
import struct
import numpy as np

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--dataset', type=str, default='Mnist')
parser.add_argument('--dataset_path', type=str, default="../data/",
                    help='Dataset path.')
parser.add_argument('--output_path', type=str, default="./preprocess_Result",
                    help='preprocess Result path.')
parser.add_argument('--latent_path', type=str, default="../data/input/",
                    help='input latent path.')
args_opt = parser.parse_args()

# train set
train_images_idx3_ubyte_file = args_opt.dataset_path + 'train-images.idx3-ubyte'
# train labels set
train_labels_idx1_ubyte_file = args_opt.dataset_path + 'train-labels.idx1-ubyte'

# test set
test_images_idx3_ubyte_file = args_opt.dataset_path + '/t10k-images.idx3-ubyte'
# test labels set
test_labels_idx1_ubyte_file = args_opt.dataset_path + 't10k-labels.idx1-ubyte'

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


def load_test_data():
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    return test_images

def load_valid_data():
    valid_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    return valid_images[-10000:]

if __name__ == '__main__':
    test_data = load_test_data()
    test_image = np.array(test_data).astype(np.float32)
    valid_data = load_valid_data()
    valid_image = np.array(valid_data).astype(np.float32)
    valid_image = valid_image.reshape((valid_image.shape[0], 784))
    test_image = test_image.reshape((test_image.shape[0], 784))
    valid_name = os.path.join(args_opt.output_path, "valid_data.txt")
    test_name = os.path.join(args_opt.output_path, "test_data.txt")
    if not os.path.exists(args_opt.output_path):
        os.makedirs(args_opt.output_path)
    if not os.path.exists(args_opt.latent_path):
        os.makedirs(args_opt.latent_path)
    np.savetxt(valid_name, (valid_image))
    np.savetxt(test_name, (test_image))
    for i in range(10000):
        input_latent = np.random.normal(size=(1, 100)).astype(np.float32)
        name = os.path.join(args_opt.latent_path, "input_latent" + str(i) + ".bin")
        input_latent.tofile(name)
