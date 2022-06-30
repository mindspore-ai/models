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
Produce the dataset
"""

import os

import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype

def create_dataset_cifar10(data_path, batch_size=32, num_parallel_workers=8, do_train=True):
    """
    create cifar10 dataset for train or test
    """
    # define dataset
    data_path = os.path.join(data_path, "cifar-10-batches-bin" if do_train else "cifar-10-verify-bin")

    cifar_ds = ds.Cifar10Dataset(data_path, num_parallel_workers=num_parallel_workers, shuffle=do_train)

    # define map operations
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
    random_horizontal_op = CV.RandomHorizontalFlip(prob=0.5)
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    if do_train:
        compose_op = [random_crop_op, random_horizontal_op, resize_op, rescale_op, normalize_op, hwc2chw_op]
    else:
        compose_op = [resize_op, rescale_op, normalize_op, hwc2chw_op]
    cifar_ds = cifar_ds.map(input_columns="image", operations=compose_op, num_parallel_workers=num_parallel_workers)
    cifar_ds = cifar_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)

    return cifar_ds
