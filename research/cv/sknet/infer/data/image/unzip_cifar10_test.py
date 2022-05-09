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
import argparse
import os
import struct
import cv2
import numpy as np


def _read_a_label(file_object):
    raw_label = file_object.read(1)
    label = struct.unpack(">B", raw_label)
    return label


def _read_a_image(file_object):
    raw_img = file_object.read(32 * 32)
    red_img = struct.unpack(">1024B", raw_img)

    raw_img = file_object.read(32 * 32)
    green_img = struct.unpack(">1024B", raw_img)

    raw_img = file_object.read(32 * 32)
    blue_img = struct.unpack(">1024B", raw_img)

    img = np.zeros(shape=(1024, 3))
    for i in range(1024):
        l = [red_img[i], green_img[i], blue_img[i]]
        img[i] = l
    img = np.reshape(img, [32, 32, 3])
    return img


def save_image(image, file_path):
    cv2.imwrite(file_path, image)


def unzip(dataset_path, result_path):
    if not os.path.exists(dataset_path):
        print(f"not such file:{dataset_path}")
        exit(-1)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if os.path.exists("label.txt"):
        os.remove("label.txt")
    with open(dataset_path, "rb") as file:
        count = 1
        while True:
            try:
                label = _read_a_label(file)
                image = _read_a_image(file)
                filename = "img" + str(count)
                save_image(image, os.path.join(result_path, filename + ".png"))
                print(f"{filename} saved")
                with open("label.txt", "a+") as f:
                    f.write(filename + " " + str(label[0]) + "\n")

            except IOError as err:
                print(err)
                break
            count += 1
            if count > 10000:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="unzip cifar10 test")
    parser.add_argument('--dataset_path', type=str, default='./test_batch.bin', help='path of file "test_batch.bin"')
    parser.add_argument('--result_path', type=str, default='./test_batch', help='path of folder to save unzip images')
    args_opt = parser.parse_args()
    unzip(args_opt.dataset_path, args_opt.result_path)
