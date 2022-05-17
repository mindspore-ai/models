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
import numpy as np

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset, cifar10, imagenet2012')
parser.add_argument('--dataset_path', type=str, default="../../../cifar10/val/",
                    help='Dataset path.')
parser.add_argument('--output_path', type=str, default="./preprocess_Result",
                    help='preprocess Result path.')
args_opt = parser.parse_args()

LABEL_SIZE = 1
IMAGE_SIZE = 32
NUM_CHANNELS = 3
TEST_NUM = 10000

def extract_data(filenames):
    '''extract_data'''

    labels = None
    images = None

    for f in filenames:
        bytestream = open(f, 'rb')
        buf = bytestream.read(TEST_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_SIZE))

        data = np.frombuffer(buf, dtype=np.uint8)

        data = data.reshape(TEST_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)

        labels_images = np.hsplit(data, [LABEL_SIZE])

        label = labels_images[0].reshape(
            TEST_NUM)

        image = labels_images[1].reshape(TEST_NUM, IMAGE_SIZE, IMAGE_SIZE,
                                         NUM_CHANNELS)

        if not labels:
            labels = label
            images = image
        else:
            labels = np.concatenate((labels, label))
            images = np.concatenate((images, image))

    return labels, images.astype(np.float32)

def extract_test_data(files_dir):
    '''extract cifar files.'''
    filenames = [os.path.join(files_dir, 'test_batch.bin'),]
    return extract_data(filenames)




def get_cifar_bin():
    '''generate cifar bin files.'''
    batch_size = 1

    labels, images = extract_test_data(args_opt.dataset_path)
    images = images / 255.0
    a = [0.4914, 0.4824, 0.4467]
    b = [0.2471, 0.2435, 0.2616]

    images = images.reshape((TEST_NUM, NUM_CHANNELS, IMAGE_SIZE,
                             IMAGE_SIZE))
    images[:, 0, :, :] = (images[:, 0, :, :] - a[0]) / b[0]
    images[:, 1, :, :] = (images[:, 1, :, :] - a[1]) / b[1]
    images[:, 2, :, :] = (images[:, 2, :, :] - a[2]) / b[2]


    img_path = os.path.join(args_opt.output_path, "00_img_data")
    label_path = os.path.join(args_opt.output_path, "label.txt")
    os.makedirs(img_path)
    label_list = []

    for i in range(TEST_NUM):
        img_data = images[i]
        img_label = labels[i]
        file_name = args_opt.dataset + "_bs" + str(batch_size) + "_" + str(i) + ".bin"

        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)
        label_list.append(file_name + ',' + str(img_label))
    np.savetxt(label_path, label_list, fmt="%s")
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    get_cifar_bin()
