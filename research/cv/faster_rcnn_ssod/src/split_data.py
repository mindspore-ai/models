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

import shutil
import random
import os

random.seed(10)
image_original_path = '/home/lwx1090586/datasets/FaceMaskDetectionDataset/images/'
label_original_path = '/home/lwx1090586/datasets/FaceMaskDetectionDataset/annotations/'
train_image_path = '/home/lwx1090586/datasets/FaceMaskDetectionDataset/train/images/'
train_label_path = '/home/lwx1090586/datasets/FaceMaskDetectionDataset/train/annotations/'
val_image_path = '/home/lwx1090586/datasets/FaceMaskDetectionDataset/val/images/'
val_label_path = '/home/lwx1090586/datasets/FaceMaskDetectionDataset/val/annotations/'


train_percent = 0.8
val_percent = 0.2

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

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    print(num_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)

    train = random.sample(list_all_txt, num_train)
    val_test = [i for i in list_all_txt if not i in train]
    val = random.sample(val_test, num_val)
    print(f'train set number:{len(train)}, val set number:{len(val)}')
    for i in list_all_txt:
        name = total_txt[i][:-4]

        srcImage = image_original_path + name + '.png'
        srcLabel = label_original_path + name + '.xml'

        if i in train:
            dst_train_Image = train_image_path + name + '.png'
            dst_train_Label = train_label_path + name + '.xml'
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        elif i in val:
            dst_val_Image = val_image_path + name + '.png'
            dst_val_Label = val_label_path + name + '.xml'
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)


if __name__ == '__main__':
    main()
