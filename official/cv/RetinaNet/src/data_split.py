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

"""data_split"""

import os
import shutil

image_original_path = '../images/'
label_original_path = '../annotations/'

train_image_path = '../dataset/train/images/'
train_label_path = '../dataset/train/annotations/'

val_image_path = '../dataset/val/images/'
val_label_path = '../dataset/val/annotations/'


def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)


def main():
    mkdir()
    with open("./data/facemask/train.txt", 'r') as f:
        for line in f:
            dst_train_image = train_image_path + line[:-1] + '.jpg'
            dst_train_label = train_label_path + line[:-1] + '.xml'
            shutil.copyfile(image_original_path + line[:-1] + '.png', dst_train_image)
            shutil.copyfile(label_original_path + line[:-1] + '.xml', dst_train_label)

    with open("./data/facemask/val.txt", 'r') as f:
        for line in f:
            dst_val_image = val_image_path + line[:-1] + '.jpg'
            dst_val_label = val_label_path + line[:-1] + '.xml'
            shutil.copyfile(image_original_path + line[:-1] + '.png', dst_val_image)
            shutil.copyfile(label_original_path + line[:-1] + ".xml", dst_val_label)


if __name__ == '__main__':
    main()
