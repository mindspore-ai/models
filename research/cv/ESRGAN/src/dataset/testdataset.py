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

"""ESRGAN test dataset."""

import mindspore.dataset as ds
from src.util.util import paired_paths_from_folder, get, imfrombytes, img2tensor


class TestDataset:
    """Import dataset"""
    def __init__(self, LR_path, GT_path):
        """init"""
        self.gt_folder = GT_path
        self.lq_folder = LR_path
        self.paths = paired_paths_from_folder(self.lq_folder, self.gt_folder)

    def __getitem__(self, index):
        """getitem"""
        img_item = {}
        gt_path = self.paths[index]['gt_path']
        img_bytes = get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True)

        img_item['LR'] = img_lq
        img_item['HR'] = img_gt

        return img_item['LR'], img_item['HR']

    def __len__(self):
        """length"""
        return len(self.paths)


def create_testdataset(batch_size, LR_path, GT_path):
    """create testdataset"""
    dataset = TestDataset(LR_path, GT_path)
    DS = ds.GeneratorDataset(dataset, column_names=["LR", "HR"], shuffle=False)
    DS = DS.batch(batch_size)
    return DS
