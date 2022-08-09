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
create train or eval dataset.
"""
import mindspore as ms
import mindspore.dataset as ds


def create_dataset(dataset_path, do_train, config, device_target, batch_size=32, run_distribute=False):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    if device_target == "GPU":
        if do_train:
            if run_distribute:
                from mindspore.communication.management import get_rank, get_group_size
                data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                                 num_shards=get_group_size(), shard_id=get_rank())
            else:
                data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        raise ValueError("Unsupported device_target.")

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    decode_op = ds.vision.Decode()
    resize_crop_op = ds.vision.RandomCropDecodeResize(resize_height,
                                                      scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = ds.vision.RandomHorizontalFlip(prob=0.5)

    resize_op = ds.vision.Resize(256)
    center_crop = ds.vision.CenterCrop(resize_width)
    rescale_op = ds.vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    # Computed from random subset of ImageNet training images
    normalize_op = ds.vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                       std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = ds.vision.HWC2CHW()

    if do_train:
        trans = [resize_crop_op, horizontal_flip_op, rescale_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set


def create_dataset_cifar(dataset_path,
                         do_train,
                         batch_size=32):
    """
    create a train or evaluate cifar10 dataset
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    data_set = ds.Cifar10Dataset(dataset_path,
                                 num_parallel_workers=8,
                                 shuffle=True)
    # define map operations
    if do_train:
        trans = [
            ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            ds.vision.RandomHorizontalFlip(prob=0.5),
            ds.vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            ds.vision.Resize((224, 224)),
            ds.vision.Rescale(1.0 / 255.0, 0.0),
            ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ds.vision.CutOut(112),
            ds.vision.HWC2CHW()
        ]
    else:
        trans = [
            ds.vision.Resize((224, 224)),
            ds.vision.Rescale(1.0 / 255.0, 0.0),
            ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ds.vision.HWC2CHW()
        ]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=type_cast_op,
                            input_columns="label",
                            num_parallel_workers=8)
    data_set = data_set.map(operations=trans,
                            input_columns="image",
                            num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set
