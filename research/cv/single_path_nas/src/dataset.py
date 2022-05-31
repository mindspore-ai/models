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
Data operations, will be used in train.py and eval.py
"""

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision

from src.config import imagenet_cfg


def create_dataset_imagenet(dataset_path, repeat_num=1, training=True,
                            num_parallel_workers=None, shuffle=True,
                            device_num=None, rank_id=None, drop_reminder=False):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        repeat_num(int): the repeat times of dataset. Default: 1
        training(bool): whether dataset is used for train or eval. Default: True.
        num_parallel_workers(int): Number of parallel workers. Default: None.
        shuffle(bool): whether dataset is used for train or eval. Default: True.
        device_num(int): Number of devices for the distributed training. Default: None
        rank_id(int): Rank of the process for the distributed training. Default: None
        drop_reminder (bool): Drop reminder of the dataset,
            if its size is less than the specified batch size. Default: False
    Returns:
        dataset
    """

    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers,
                                     shuffle=shuffle, num_shards=device_num, shard_id=rank_id)

    assert imagenet_cfg.image_height == imagenet_cfg.image_width, "image_height not equal image_width"
    image_size = imagenet_cfg.image_height
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if training:
        transform_img = [
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomColorAdjust(0.5, 0.4, 0.3, 0.2),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        transform_img = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = [C.TypeCast(mstype.int32)]
    data_set = data_set.map(input_columns="image", num_parallel_workers=16,
                            operations=transform_img, python_multiprocessing=True)
    data_set = data_set.map(input_columns="label", num_parallel_workers=4,
                            operations=transform_label)
    # apply batch operations
    data_set = data_set.batch(imagenet_cfg.batch_size, drop_remainder=drop_reminder)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
