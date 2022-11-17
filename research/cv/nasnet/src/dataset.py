# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C2
import mindspore.dataset.vision as C

def create_dataset(dataset_path, do_train, rank, group_size,
                   num_parallel_workers=8, batch_size=128,
                   drop_remainder=True, shuffle=True,
                   cutout=False, cutout_length=56, image_size=224,
                   repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        rank(int): The shard ID within num_shards (default=None).
        group_size(int): Number of shards that the dataset should be divided into (default=None).
        num_parallel_workers(int): the number of parallel workers (Default:8).
        batch_size(int): the batch size for dataset (Default:128).
        drop_remainder(bool): whether to drop the remainder in dataset (Default:True).
        shuffle(bool): whether to shuffle the dataset (Default:True).
        cutout(bool): whether to cutout the data during trainning (Default:False).
        cutout_length(int): the length to cutout data when cutout is True (Default:56).
        image_size(int): the size of image (Default:224).
        repeat_num(int): the repeat times of dataset (Default:1).

    Returns:
        dataset
    """
    device_target = mindspore.context.get_context("device_target")
    if group_size == 1  or not do_train:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers,
                                         shuffle=shuffle)
        print(dataset_path)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers,
                                         shuffle=shuffle,
                                         num_shards=group_size, shard_id=rank)
        print(dataset_path, ' group_size = ', group_size, ' rank = ', rank)
    # define transform operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size),
        ]
        if cutout:
            trans += [C.CutOut(length=cutout_length, num_patches=1)]
        trans += [
            C.RandomHorizontalFlip(prob=0.5),
        ]

        if device_target == 'GPU':
            trans += [C.RandomColorAdjust(brightness=0.4, contrast=0.5)]
        else:
            trans += [C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)]
    else:
        trans = [
            C.Decode(),
            C.Resize(round(image_size / 0.875)),
            C.CenterCrop(image_size)
        ]

    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        C.HWC2CHW(),
        C2.TypeCast(mstype.float32)
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image",
                            num_parallel_workers=num_parallel_workers)
    data_set = data_set.map(operations=type_cast_op, input_columns="label",
                            num_parallel_workers=num_parallel_workers)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=drop_remainder)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    return data_set
