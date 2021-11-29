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
"""data prepare for CUB200-2011"""
import os
import shutil
import time

path = './'

ROOT_TRAIN = path + 'images/train/'
ROOT_TEST = path + 'images/test/'
BATCH_SIZE = 16

time_start = time.time()

path_images = path + 'images.txt'
path_split = path + 'train_test_split.txt'
trian_save_path = path + 'dataset/train/'
test_save_path = path + 'dataset/test/'

images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))

split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split(',')))

num = len(images)
for k in range(num):
    file_name = images[k][0].split(' ')[1].split('/')[0]
    aaa = int(split[k][0][-1])
    if int(split[k][0][-1]) == 1:
        if os.path.isdir(trian_save_path + file_name):
            shutil.copy(path + 'images/' + images[k][0].split(' ')[1],
                        trian_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        else:
            os.makedirs(trian_save_path + file_name)
            shutil.copy(path + 'images/' + images[k][0].split(' ')[1],
                        trian_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        print('%s finished!' % images[k][0].split(' ')[1].split('/')[1])
    else:
        if os.path.isdir(test_save_path + file_name):
            aaaa = path + 'images/' + images[k][0].split(' ')[1]
            bbbb = test_save_path + file_name + '/' + images[k][0].split(' ')[1]
            shutil.copy(path + 'images/' + images[k][0].split(' ')[1],
                        test_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        else:
            os.makedirs(test_save_path + file_name)
            shutil.copy(path + 'images/' + images[k][0].split(' ')[1],
                        test_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        print('%s finished!' % images[k][0].split(' ')[1].split('/')[1])

time_end = time.time()
print('CUB200 finished, time consume %s!!' % (time_end - time_start))
