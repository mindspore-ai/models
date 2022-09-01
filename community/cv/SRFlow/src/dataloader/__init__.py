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
"""
The Dataloader of Train and Validation
"""

import mindspore.dataset as ds
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.dataset.vision.py_transforms as py_vision


def create_train_dataset(opt, group_size, rank_id):
    """
    Create the train dataset
    """
    dataroot = opt['datasets']['train']['dataroot']
    batch_size = opt['datasets']['train']['batch_size']
    dataset = ds.MindDataset(dataset_files=dataroot, shard_id=rank_id, num_shards=group_size)

    hr_transforms = Compose([py_vision.ToTensor()])
    lr_transforms = Compose([py_vision.ToTensor()])
    dataset = dataset.map(operations=hr_transforms, input_columns='HR')
    dataset = dataset.map(operations=lr_transforms, input_columns='LR')

    dataset = dataset.batch(batch_size)

    return dataset


def create_valid_dataset(opt):
    """
    Create the valid dataset
    """
    dataroot = opt['datasets']['val']['dataroot']
    dataset = ds.MindDataset(dataset_files=dataroot)

    hr_transforms = Compose([py_vision.ToTensor()])
    lr_transforms = Compose([py_vision.ToTensor()])
    dataset = dataset.map(operations=hr_transforms, input_columns='HR')
    dataset = dataset.map(operations=lr_transforms, input_columns='LR')

    dataset = dataset.batch(1)

    return dataset
