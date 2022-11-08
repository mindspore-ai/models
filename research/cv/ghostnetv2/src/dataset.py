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
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, do_train, infer_910=False, device_id=0, batch_size=128, num_parallel_workers=8):
    """
    create a train or eval dataset

    Args:
        dataset_path (string): The path of dataset.
        do_train (bool): Whether dataset is used for train or eval.
        infer_910 (bool): Whether to use Ascend 910.
        device_id (int): Device id.
        batch_size (int): Input image batch size.
        num_parallel_workers (int): Number of workers to read the data.

    Returns:
        dataset
    """
    device_num = 1
    device_id = device_id
    rank_id = os.getenv('RANK_ID', '0')
    if infer_910:
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))

    if not do_train:
        dataset_path = os.path.join(dataset_path)
    else:
        dataset_path = os.path.join(dataset_path)

    if device_num == 1:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=True)
    else:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(224),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(224),
        ]
    trans += [
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW(),
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=num_parallel_workers)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
