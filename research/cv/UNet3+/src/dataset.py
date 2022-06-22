# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
'''data loader'''
import os
import glob
import random
import numpy as np
from skimage.io import imread

import mindspore.dataset as ds
import mindspore.dataset.vision as CV

class Dataset:
    '''Dataset'''
    def __init__(self, data_path, aug=False):
        super(Dataset, self).__init__()
        self.img_paths = glob.glob(os.path.join(data_path, "ct", "*"))
        self.mask_paths = glob.glob(os.path.join(data_path, "seg", "*"))
        self.aug = aug

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        mask = mask[:, :, np.newaxis]
        return image, mask

    def __len__(self):
        return len(self.img_paths)


def create_Dataset(data_path, aug, batch_size, device_num, rank, shuffle):

    dataset = Dataset(data_path, aug)
    hwc_to_chw = CV.HWC2CHW()
    data_set = ds.GeneratorDataset(dataset, column_names=["image", "mask"], \
               num_parallel_workers=8, shuffle=shuffle, num_shards=device_num, shard_id=rank)
    data_set = data_set.map(input_columns=["image"], operations=hwc_to_chw, num_parallel_workers=8)
    data_set = data_set.map(input_columns=["mask"], operations=hwc_to_chw, num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, data_set.get_dataset_size()
