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
"""Data operations, will be used in train.py and eval.py"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms as C2

from mindspore.dataset.transforms.transforms import Compose
import mindspore.dataset.vision as C


def create_dataset(dataset_path, do_train, repeat_num=1, infer_910=True, device_id=0, batch_size=128):
    """
    create a train or eval dataset

    Args:
        batch_size:
        device_id:
        infer_910:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """

    if not do_train:
        dataset_path = os.path.join(dataset_path, 'val')
    else:
        dataset_path = os.path.join(dataset_path, 'train')

    ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True, num_shards=1, shard_id=0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(224),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
        ]
    else:
        trans = [
            C.Decode(True),
            C.Resize(235),
            C.CenterCrop(224)
        ]
    trans += [
        C.ToTensor(),
        C.Normalize(mean=mean, std=std, is_hwc=False),
    ]
    trans = Compose(trans)

    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=8)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_workers=8)
    ds = ds.repeat(repeat_num)
    return ds
