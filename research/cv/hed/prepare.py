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
'''BSDS500'''
import os
import errno
import imageio
from scipy import io
import numpy as np
from src.model_utils.config import config
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

GT_PATH = os.path.join(os.path.join(config.data_path, 'BSDS500/data/groundTruth'))
label_train_path = os.path.join(config.data_path, 'BSDS500/data/labels/train/')
if not os.path.isdir(label_train_path):
    os.makedirs(label_train_path)
train_lst_path = os.path.join(os.path.join(config.data_path, 'output/train.lst'))
if not os.path.isdir(train_lst_path):
    mkdir_p(os.path.dirname(train_lst_path))
test_lst_path = os.path.join(os.path.join(config.data_path, 'output/test.lst'))
if not os.path.isdir(test_lst_path):
    mkdir_p(os.path.dirname(test_lst_path))
val_lst_path = os.path.join(os.path.join(config.data_path, 'output/val.lst'))
if not os.path.isdir(val_lst_path):
    mkdir_p(os.path.dirname(val_lst_path))

train_list = os.listdir(GT_PATH+'/train/')
print(len(train_list))
for index in train_list:
    name = index.split('.')[0]
    print(name)
    train = io.loadmat(GT_PATH+'/train/'+index)
    a = np.array(1024)
    a = train['groundTruth'][0][0][0][0][1]
    print(a)
    a = a*255
    print(label_train_path+str(name))
    imageio.imsave(label_train_path+str(name)+'.jpg', a)

label_test_path = os.path.join(config.data_path, 'BSDS500/data/labels/test/')
if not os.path.isdir(label_test_path):
    os.makedirs(label_test_path)

test_list = os.listdir(GT_PATH+'/test/')
print(len(test_list))
for index in test_list:
    name = index.split('.')[0]
    print(name)
    test = io.loadmat(GT_PATH+'/test/'+index)
    #print(train)
    a = np.array(1024)
    a = test['groundTruth'][0][0][0][0][1]
    print(a)
    a = a*255
    print(label_test_path+str(name))
    imageio.imsave(label_test_path+str(name)+'.jpg', a)

label_val_path = os.path.join(config.data_path, 'BSDS500/data/labels/val/')
if not os.path.isdir(label_val_path):
    os.makedirs(label_val_path)

val_list = os.listdir(GT_PATH+'/val/')
print(len(val_list))
for index in val_list:
    name = index.split('.')[0]
    print(name)
    val = io.loadmat(GT_PATH+'/val/'+index)
    #print(train)
    a = np.array(1024)
    a = val['groundTruth'][0][0][0][0][1]
    print(a)
    a = a*255
    print(label_val_path+str(name))
    imageio.imsave(label_val_path+str(name)+'.jpg', a)
