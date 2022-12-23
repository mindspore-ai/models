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
import os
import numpy as np

import mindspore as ms
import mindspore.dataset as ds


def create_dataset(dataset_path, do_train, config, enable_cache=False, cache_session_id=None):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        enable_cache(bool): whether tensor caching service is used for dataset on nfs. Default: False
        cache_session_id(string): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        nfs_dataset_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
    else:
        nfs_dataset_cache = None

    num_workers = config.num_workers
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers, shuffle=do_train,
                                     num_shards=config.rank_size, shard_id=config.rank_id, cache=nfs_dataset_cache)

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    decode_op = ds.vision.Decode()
    resize_crop_op = ds.vision.RandomCropDecodeResize(resize_height,
                                                      scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = ds.vision.RandomHorizontalFlip(prob=0.5)

    resize_op = ds.vision.Resize((256, 256))
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

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=num_workers)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_workers)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(config.batch_size, drop_remainder=True)

    return data_set


def create_dataset_cifar10(dataset_path, do_train, config, enable_cache=False, cache_session_id=None):
    """
    create cifar-10 train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        enable_cache(bool): whether tensor caching service is used for dataset on nfs. Default: False
        cache_session_id(string): If enable_cache, cache session_id need to be provided. Default: None

    Returns:
        dataset
    """
    num_workers = config.num_workers
    resize_height = config.image_height
    resize_width = config.image_width

    device_num = config.device_num
    if do_train:
        if device_num == 1:
            data_set = ds.Cifar10Dataset(dataset_path, usage='train', num_parallel_workers=num_workers, shuffle=True)
        else:
            data_set = ds.Cifar10Dataset(dataset_path, usage='train', num_parallel_workers=num_workers, shuffle=True,
                                         num_shards=device_num, shard_id=config.rank_id)
    else:
        if device_num == 1:
            data_set = ds.Cifar10Dataset(dataset_path, usage='test', num_parallel_workers=num_workers, shuffle=False)
        else:
            data_set = ds.Cifar10Dataset(dataset_path, usage='test', num_parallel_workers=num_workers, shuffle=False,
                                         num_shards=device_num, shard_id=config.rank_id)
    n = 0
    for _ in data_set.create_dict_iterator():
        n += 1
    print("data_num={}".format(n))
    trans = []
    if do_train:
        trans += [
            ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            ds.vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        ds.vision.Resize((resize_height, resize_width)),
        ds.vision.Rescale(1.0 / 255.0, 0.0),
        ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ds.vision.HWC2CHW()
    ]

    type_cast_op = ds.transforms.transforms.TypeCast(ms.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label",
                            num_parallel_workers=num_workers)
    # only enable cache for eval
    if do_train:
        enable_cache = False
    if enable_cache:
        if not cache_session_id:
            raise ValueError("A cache session_id must be provided to use cache.")
        eval_cache = ds.DatasetCache(session_id=int(cache_session_id), size=0)
        data_set = data_set.map(operations=trans, input_columns="image",
                                num_parallel_workers=num_workers, cache=eval_cache)
    else:
        data_set = data_set.map(operations=trans, input_columns="image",
                                num_parallel_workers=num_workers)

    # apply batch operations
    data_set = data_set.batch(config.batch_size, drop_remainder=True)

    return data_set


def extract_features(net, dataset_path, config):
    features_folder = dataset_path + '_features'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    dataset = create_dataset(dataset_path=dataset_path,
                             do_train=False,
                             config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")

    model = ms.Model(net)

    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        features_path = os.path.join(features_folder, f"feature_{i}.npy")
        label_path = os.path.join(features_folder, f"label_{i}.npy")
        if not os.path.exists(features_path) or not os.path.exists(label_path):
            image = data["image"]
            label = data["label"]
            features = model.predict(ms.Tensor(image))
            np.save(features_path, features.asnumpy())
            np.save(label_path, label)
        print(f"Complete the batch {i + 1}/{step_size}")
    return step_size
