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
Get dataset object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

from src.get_dataset import get_loader


def get_training_set(config, spatial_transform, temporal_transform, target_transform):
    assert config.dataset in ['ucf101', 'hmdb51']

    training_data = get_loader(
        config.video_path,
        config.annotation_path,
        'training',
        config.mode,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=config.train_sample_duration)

    return training_data


def get_validation_set(config, spatial_transform, temporal_transform, target_transform):
    assert config.dataset in ['ucf101', 'hmdb51']

    # Disable evaluation
    if config.no_eval:
        return None

    validation_data = get_loader(
        config.video_path,
        config.annotation_path,
        'validation',
        config.mode,
        config.num_val_samples,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=config.test_sample_duration)

    return validation_data


def get_dataset(config, train_transforms, validation_transforms=None):
    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))
    dataset = {}
    dataset_train = get_training_set(config, train_transforms['spatial'],
                                     train_transforms['temporal'], train_transforms['target'])
    print('Found {} training examples'.format(len(dataset_train)))

    if config.distributed:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset['train'] = ds.GeneratorDataset(source=dataset_train, column_names=["clip", "target"],
                                               shuffle=True, num_parallel_workers=config.num_workers,
                                               num_shards=rank_size, shard_id=rank_id)
    else:
        dataset['train'] = ds.GeneratorDataset(source=dataset_train, column_names=["clip", "target"],
                                               shuffle=True, num_parallel_workers=config.num_workers)

    dataset['train'] = dataset['train'].batch(batch_size=config.batch_size, num_parallel_workers=config.num_workers)

    if not config.no_eval and validation_transforms:

        dataset_validation = get_validation_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])
        print('Found {} validation examples'.format(len(dataset_validation)))

        if config.distributed:
            rank_id = get_rank()
            rank_size = get_group_size()
            dataset['validation'] = ds.GeneratorDataset(source=dataset_validation, column_names=["clip", "target"],
                                                        shuffle=True, num_parallel_workers=config.num_workers,
                                                        num_shards=rank_size, shard_id=rank_id, max_rowsize=18)
        else:
            dataset['validation'] = ds.GeneratorDataset(source=dataset_validation, column_names=["clip", "target"],
                                                        shuffle=True, num_parallel_workers=config.num_workers,
                                                        max_rowsize=18)
        dataset['validation'] = dataset['validation'].batch(batch_size=config.batch_size,
                                                            num_parallel_workers=config.num_workers)

    return dataset
