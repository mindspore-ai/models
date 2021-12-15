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
"""
Different from the original paper, it is an adaptation of the Mindspore version
"""
import os
import random
import glob
import h5py
import numpy as np
from src.common import crop_size

class H5Dataset:
    """
    H5Dataset used for loading h5 file eg. xxx.h5
    """
    def __init__(self, root_path, c_size=crop_size, mode='train'):
        self.hdf5_list = [x for x in glob.glob(os.path.join(root_path, '*.h5'))]
        self.crop_size = c_size
        self.mode = mode
        if self.mode == 'train':
            self.hdf5_list = self.hdf5_list + self.hdf5_list + self.hdf5_list + self.hdf5_list

    def __getitem__(self, index):
        h5_file = h5py.File(self.hdf5_list[index])
        self.data = h5_file.get('data')
        self.label = h5_file.get('label')
        self.label = self.label[:, 0, ...]
        _, _, C, H, W = self.data.shape
        if self.mode == 'train':
            cx = random.randint(0, C - self.crop_size[0])
            cy = random.randint(0, H - self.crop_size[1])
            cz = random.randint(0, W - self.crop_size[2])

        elif self.mode == 'val':
            #Center crop
            cx = (C - self.crop_size[0])//2
            cy = (H - self.crop_size[1])//2
            cz = (W - self.crop_size[2])//2

        self.data_crop = self.data[:, :, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1],\
                         cz: cz + self.crop_size[2]]
        self.label_crop = self.label[:, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], \
                          cz: cz + self.crop_size[2]]
        #End random crop
        data = self.data_crop[0, :, :, :, :]
        label = self.label_crop[0, :, :, :].astype(np.int32)
        one_hot = np.zeros((4, label.shape[0], label.shape[1], label.shape[2]),\
                           dtype=label.dtype)
        for class_id in range(4):
            one_hot[class_id, ...] = (label == class_id)
        return data, one_hot

    def __len__(self):
        return len(self.hdf5_list)
