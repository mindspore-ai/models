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

import datetime
import sys
import os
from mindspore import dataset as ds
import train.provider as provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
sys.path.append(ROOT_DIR)

BATCH_SIZE = 32
cols = [
    "data", "label", "center", "hclass", "hres", "sclass", "sres",
    "rot_angle", "one_hot_vec"
]


def get_train_data(num_samples=None, device_num=None, rank_id=None):
    print(f"{datetime.datetime.now().isoformat()}:loading train dataset ...")
    dss = provider.FrustumDataset(npoints=1024,
                                  split='train',
                                  rotate_to_center=True,
                                  random_flip=True,
                                  random_shift=True,
                                  one_hot=True)
    train_data_set = ds.GeneratorDataset(dss,
                                         cols, num_samples=num_samples, \
                                            num_shards=device_num, shard_id=rank_id, shuffle=True)
    train_data_set = train_data_set.batch(BATCH_SIZE,
                                          True,
                                          num_parallel_workers=4)
    return train_data_set


def get_test_data(device_num=None, rank_id=None):
    print(f"{datetime.datetime.now().isoformat()}:loading test dataset ...")
    # ['center', 'data', 'hclass', 'hres', 'label', 'one_hot_vec', 'rot_angle', 'sclass', 'sres']
    dss = provider.FrustumDataset(npoints=1024,
                                  split='val',
                                  rotate_to_center=True,
                                  random_flip=True,
                                  random_shift=True,
                                  one_hot=True)
    test_data_set = ds.GeneratorDataset(dss,
                                        cols, shuffle=False)
    test_data_set = test_data_set.batch(BATCH_SIZE,
                                        True,
                                        num_parallel_workers=4)

    return test_data_set
