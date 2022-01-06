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
"""haze_data.py"""
import os
from PIL import ImageFile
import imageio
from src import common

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RESIDEDatasetGenerator:
    """[RESIDEDatasetGenerator]
    """
    def __init__(self, args, train, ext='.png'):
        self.args = args
        path = args.dir_data
        self.ext = ext
        self.train = train
        if train:
            self.data_name = args.data_train[0]
            self.haze_imgs_dir = os.listdir(os.path.join(path, self.data_name, 'train', 'hazy'))
            self.haze_imgs = [os.path.join(path, self.data_name, 'train', 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, self.data_name, 'train', 'clear')
        else:
            self.data_name = args.data_test[0] # Dense-Haze
            self.haze_imgs_dir = os.listdir(os.path.join(path, self.data_name, 'hazy'))
            self.haze_imgs = [os.path.join(path, self.data_name, 'hazy', img) for img in self.haze_imgs_dir]
            self.clear_dir = os.path.join(path, self.data_name, 'GT')

    def _get_index(self, idx):
        """get_index"""
        if self.train:
            return idx % len(self.haze_imgs)
        return idx

    def _load_file(self, idx):
        """load_file"""
        idx = self._get_index(idx)
        f_hazy = self.haze_imgs[idx]
        filename = f_hazy.split('/')[-1]
        if self.data_name == 'RESIDE':
            img_id = filename.split('_')[0]
            gt_file = f"{img_id}.png"
        elif self.data_name == 'NHHaze':
            img_name = filename.split('_')
            gt_file = f"{img_name[0]}_GT.png"
        elif self.data_name == 'Dense':
            img_name = filename.split('_')
            gt_file = f"{img_name[0]}_GT.png"
        else:
            print(f"{self.data_name} Dataset not implemented...")
            return None
        gt = imageio.imread(os.path.join(self.clear_dir, gt_file))
        hazy = imageio.imread(f_hazy)
        return hazy, gt, filename

    def get_patch(self, hazy, gt):
        """get_patch"""
        if self.train:
            hazy, gt = common.get_patch(
                hazy, gt,
                patch_size=self.args.patch_size,
                scale=1)
            if not self.args.no_augment:
                hazy, gt = common.augment(hazy, gt)
        return hazy, gt

    def __getitem__(self, idx):
        """get item"""
        hazy, gt, _ = self._load_file(idx)
        pair = self.get_patch(hazy, gt)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2_tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1]

    def __len__(self):
        print("haze images:", len(self.haze_imgs))
        return len(self.haze_imgs)
