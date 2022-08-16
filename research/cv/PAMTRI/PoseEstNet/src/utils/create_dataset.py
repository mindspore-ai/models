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

import csv
import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='Create PoseEstNet dataset from VeRi')
parser.add_argument('--veri', type=str, help='The path of veri dataset')
parser.add_argument('--PoseData', type=str, help='The path of PoseEstNet Dataset')

args = parser.parse_args()

# The contents of original veri dataset
veri = args.veri

# root directory of PoseEstNet Dataset
data_root = args.PoseData

def check_path(path_name):
    '''
    Check if directory exists
    '''
    if not os.path.exists(path_name):
        os.mkdir(path_name)

# Check data_root/annot directory
check_path(os.path.join(data_root, 'annot'))
# Check data_root/images directory
check_path(os.path.join(data_root, 'images'))

image_test_path = os.path.join(data_root, 'images', 'image_test')
image_train_path = os.path.join(data_root, 'images', 'image_train')

# Check data_root/images/image_train directory
check_path(image_train_path)
# Check data_root/images/image_test directory
check_path(image_test_path)

train_csv = os.path.join(data_root, 'annot', 'label_train.csv')
test_csv = os.path.join(data_root, 'annot', 'label_test.csv')

# build data_root/images/image_train dataset
with open(train_csv, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        image_name = row[0]
        image_veri = os.path.join(veri, 'image_train')
        src = os.path.join(image_veri, image_name)
        dst = os.path.join(image_train_path, image_name)
        shutil.copy(src, dst)

# build data_root/images/image_test dataset
with open(test_csv, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        image_name = row[0]
        image_veri = os.path.join(veri, 'image_train')
        src = os.path.join(image_veri, image_name)
        dst = os.path.join(image_test_path, image_name)
        shutil.copy(src, dst)


print(f'Size of {image_train_path} dataset：', len(os.listdir(image_train_path)))
print(f'Size of {image_test_path} dataset：', len(os.listdir(image_test_path)))
