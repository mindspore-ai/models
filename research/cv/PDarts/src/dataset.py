# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Read train and eval data"""
import mindspore.dataset as ds
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter


def create_cifar10_dataset(data_dir, training=True, repeat_num=1, num_parallel_workers=5,
                           resize_height=32, resize_width=32, batch_size=512,
                           num_samples=None, shuffle=None, cutout_length=0, rank_id=0, rank_size=1):
    """Data operations."""
    ds.config.set_seed(1)
    ds.config.set_num_parallel_workers(num_parallel_workers)

    if training:
        data_set = ds.Cifar10Dataset(data_dir, num_samples=num_samples,
                                     shuffle=shuffle, num_shards=rank_size, shard_id=rank_id)
    else:
        data_set = ds.Cifar10Dataset(data_dir, num_samples=num_samples,
                                     shuffle=shuffle, num_shards=1, shard_id=0)

    # define map operations
    random_crop_op = vision.RandomCrop(
        (resize_height, resize_width), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    random_rotation = vision.RandomRotation(5, resample=Inter.BICUBIC)
    # resize_op = vision.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    normalize_op = vision.Normalize((125.3, 123.0, 113.9), (63.0, 62.1, 66.7))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op, random_rotation]
        if cutout_length > 0:
            random_cutout = vision.CutOut(cutout_length)
            c_trans += [random_cutout]
    c_trans += [normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(input_columns="label", operations=type_cast_op,
                            num_parallel_workers=num_parallel_workers)
    data_set = data_set.map(input_columns="image", operations=c_trans,
                            num_parallel_workers=num_parallel_workers)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=1000)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
