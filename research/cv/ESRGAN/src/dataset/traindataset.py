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

"""ESRGAN train dataset."""

import mindspore.dataset as ds
from src.util.util import paired_paths_from_folder, get, imfrombytes, paired_random_crop, augment, img2tensor


class TrainDataset:
    """Import dataset"""
    def __init__(self, LR_path, GT_path, scale, image_size):
        """init"""
        self.gt_folder = GT_path
        self.lq_folder = LR_path

        self.scale = scale
        self.image_size = image_size
        self.paths = paired_paths_from_folder(self.lq_folder, self.gt_folder)

    def __getitem__(self, index):
        """getitem"""
        img_item = {}
        scale = self.scale
        gt_path = self.paths[index]['gt_path']
        img_bytes = get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)

        image_size = self.image_size
        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, image_size, scale, gt_path)
        # flip, rotation
        img_gt, img_lq = augment([img_gt, img_lq], True, True)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True)

        img_item['LR'] = img_lq
        img_item['HR'] = img_gt

        return img_item['LR'], img_item['HR']

    def __len__(self):
        """getlength"""
        return len(self.paths)


def create_traindataset(batch_size, LR_path, GT_path, scale, image_size, num_shards, shard_id):
    """"create ESRGAN dataset"""
    dataset = TrainDataset(LR_path, GT_path, scale, image_size)
    DS = ds.GeneratorDataset(dataset, column_names=['LR', 'HR'], shuffle=True,
                             num_shards=num_shards, shard_id=shard_id)
    DS = DS.batch(batch_size, drop_remainder=True)
    return DS
