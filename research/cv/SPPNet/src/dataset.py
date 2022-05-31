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
"""
Produce the dataset
"""

import os
import mindspore.dataset as ds
import mindspore.dataset.vision as CV
from mindspore.communication.management import get_rank, get_group_size


def create_dataset_imagenet(dataset_path, train_model_name='sppnet_single',
                            batch_size=256, training=True,
                            num_samples=None, workers=12,
                            shuffle=None, class_indexing=None,
                            sampler=None, image_size=224):
    """
    create a train or eval imagenet2012 dataset for Sppnet

    Args:
        dataset_path(string): the path of dataset.
        train_model_name(string): model name for training
        training(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 128
        target(str): the device target. Default: Ascend
    Returns:
        dataset
    """

    rank_size = int(os.environ.get("RANK_SIZE", 1))
    num_parallel_workers = workers

    if rank_size > 1:
        device_num = get_group_size()
        rank_id = get_rank()
    else:
        device_num = 1
        rank_id = 0

    if device_num == 1:
        num_parallel_workers = 16
        ds.config.set_prefetch_size(8)
    else:
        ds.config.set_numa_enable(True)
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=4,
                                     num_samples=num_samples, shuffle=shuffle,
                                     sampler=sampler, class_indexing=class_indexing,
                                     num_shards=device_num, shard_id=rank_id)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if training and image_size == 224:
        if train_model_name == 'zfnet':
            transform_img = [
                CV.RandomCropDecodeResize((224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                CV.RandomHorizontalFlip(prob=0.5),
                CV.Normalize(mean=mean, std=std),
                CV.HWC2CHW()
            ]
        else:
            transform_img = [
                CV.RandomCropDecodeResize((224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                CV.RandomHorizontalFlip(prob=0.5),
                CV.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
                CV.Normalize(mean=mean, std=std),
                CV.HWC2CHW()
            ]
    elif training and image_size == 180:
        transform_img = [
            CV.RandomCropDecodeResize((224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.Resize(180),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        transform_img = [
            CV.Decode(),
            CV.Resize((256, 256)),
            CV.CenterCrop(224),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]

    data_set = data_set.map(input_columns="image", operations=transform_img,
                            num_parallel_workers=num_parallel_workers)
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set
