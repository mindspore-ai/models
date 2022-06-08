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
"""Create train or eval dataset."""
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2


def create_dataset(dataset_path, do_train, img_shape, repeat_num=1, batch_size=32, run_distribute=False):
    """
    create a train or eval dataset
    """

    if do_train:
        if run_distribute:
            from mindspore.communication.management import get_rank, get_group_size
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                             num_shards=get_group_size(), shard_id=get_rank())
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)


    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

    # define map operations
    if do_train:
        trans = [
            C.Decode(),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.CenterCrop(img_shape),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
