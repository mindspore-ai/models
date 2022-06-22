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

import numpy as np

import mindspore
from mindspore import dataset as ds
from mindspore.dataset.transforms import c_transforms
from mindspore.dataset.vision import c_transforms as CV

from src.dataset.base_dataset import BaseDataset
from src import config as cfg


class ITWDataset:
    """Mixed dataset with data only from "in-the-wild" datasets (no data from H36M)."""

    def __init__(self, options):
        self.lsp_dataset = BaseDataset(options, 'lsp-orig')
        self.coco_dataset = BaseDataset(options, 'coco')
        self.mpii_dataset = BaseDataset(options, 'mpii')
        self.up3d_dataset = BaseDataset(options, 'up-3d')
        self.length = max(len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        self.partition = np.array([.1, .3, .3, .3]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        if p <= self.partition[1]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        if p <= self.partition[2]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length


def create_train_dataset(options, device_num=1, device_id=0):
    """
    Create user-defined mindspore dataset for training
    :param options: Configures for dataset
    :return: Mindspore training dataset
    """
    column_names = ['img', 'pose', 'betas', 'pose_3d', 'keypoints', 'has_smpl', 'has_pose_3d']
    type_cast_op = c_transforms.TypeCast(mindspore.float32)
    normalize_op = CV.Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    transform_list = [type_cast_op, normalize_op, CV.HWC2CHW()]
    dataset_generator = ds.GeneratorDataset(ITWDataset(options), column_names=column_names, shuffle=options.do_shuffle,
                                            num_parallel_workers=options.num_workers, num_shards=device_num,\
                                                shard_id=device_id)
    dataset_generator = dataset_generator.map(operations=transform_list, input_columns=['img'], num_parallel_workers=8)
    if options.do_shuffle:
        dataset_generator = dataset_generator.shuffle(100)
    dataset_generator = dataset_generator.batch(options.batch_size, num_parallel_workers=8, drop_remainder=True)
    return dataset_generator


def create_eval_dataset(options, dataset_name, batch_size=32, shuffle=False, num_workers=8):
    """
    Create user-defined mindspore dataset for evaluation
    dataset_name: 'up-3d' and 'lsp'
    """
    column_names = ['img', 'pose', 'betas', 'center', 'scale', 'orig_shape', 'maskname', 'partname']

    type_cast_op = c_transforms.TypeCast(mindspore.float32)
    normalize_op = CV.Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    transform_list = [type_cast_op, normalize_op, CV.HWC2CHW()]

    dataset_generator = ds.GeneratorDataset(BaseDataset(options, dataset_name, is_train=False),
                                            shuffle=shuffle,
                                            num_parallel_workers=num_workers, column_names=column_names)
    dataset_generator = dataset_generator.map(operations=transform_list, input_columns=['img'],
                                              num_parallel_workers=8)
    dataset_generator = dataset_generator.batch(batch_size, num_parallel_workers=8, drop_remainder=True)
    return dataset_generator
