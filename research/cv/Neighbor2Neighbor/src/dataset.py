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
'''dataloader'''
import os
import glob
import numpy as np
import PIL.Image as Image
import mindspore.dataset as ds
import mindspore.dataset.vision as CV

class DataLoader_Imagenet_val:
    '''DataLoader_Imagenet_val'''
    def __init__(self, data_dir, patch=256, noise_style="gauss25", batch_size=4):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))
        self.noise_generator = AugmentNoise(noise_style)
        self.batch_size = batch_size
        self.test = 1
    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        im /= 255.0 #clean image
        noisy = self.noise_generator.add_noise(im)

        return im, noisy

    def __len__(self):
        return len(self.train_fns)

class AugmentNoise():
    '''AugmentNoise'''
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_noise(self, x):
        '''add_noise'''
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        if self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        if self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        assert self.style == "poisson_range"
        min_lam, max_lam = self.params
        lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
        return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


def create_Dataset(data_dir, patch, noise_style, batch_size, device_num, rank, shuffle):

    dataset = DataLoader_Imagenet_val(data_dir, patch, noise_style, batch_size)
    hwc_to_chw = CV.HWC2CHW()
    data_set = ds.GeneratorDataset(dataset, column_names=["image", "noisy"], \
               num_parallel_workers=8, shuffle=shuffle, num_shards=device_num, shard_id=rank)
    data_set = data_set.map(input_columns=["image"], operations=hwc_to_chw, num_parallel_workers=8)
    data_set = data_set.map(input_columns=["noisy"], operations=hwc_to_chw, num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, data_set.get_dataset_size()
