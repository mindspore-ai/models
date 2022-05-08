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
import math
import os

import numpy as np
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset.vision import Inter

# values that should remain constant
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485*255, 0.456*255, 0.406*255)
IMAGENET_DEFAULT_STD = (0.229*255, 0.224*255, 0.225*255)

# data preprocess configs
SCALE = (0.08, 1.0)
RATIO = (3./4., 4./3.)

ds.config.set_seed(1)

# define Auto Augmentation operators
PARAMETER_MAX = 10

def float_parameter(level, maxval):
    return float(level) * maxval /  PARAMETER_MAX

def int_parameter(level, maxval):
    return int(level * maxval / PARAMETER_MAX)

def shear_x(level):
    v = float_parameter(level, 0.3)
    return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, shear=(-v, -v)),
                                      c_vision.RandomAffine(degrees=0, shear=(v, v))])

def shear_y(level):
    v = float_parameter(level, 0.3)
    return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, shear=(0, 0, -v, -v)),
                                      c_vision.RandomAffine(degrees=0, shear=(0, 0, v, v))])

def translate_x(level):
    v = float_parameter(level, 150 / 331)
    return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, translate=(-v, -v)),
                                      c_vision.RandomAffine(degrees=0, translate=(v, v))])

def translate_y(level):
    v = float_parameter(level, 150 / 331)
    return c_transforms.RandomChoice([c_vision.RandomAffine(degrees=0, translate=(0, 0, -v, -v)),
                                      c_vision.RandomAffine(degrees=0, translate=(0, 0, v, v))])

def color_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return c_vision.RandomColor(degrees=(v, v))

def rotate_impl(level):
    v = int_parameter(level, 30)
    return c_transforms.RandomChoice([c_vision.RandomRotation(degrees=(-v, -v)),
                                      c_vision.RandomRotation(degrees=(v, v))])

def solarize_impl(level):
    level = int_parameter(level, 256)
    v = 256 - level
    return c_vision.RandomSolarize(threshold=(0, v))

def posterize_impl(level):
    level = int_parameter(level, 4)
    v = 4 - level
    return c_vision.RandomPosterize(bits=(v, v))

def contrast_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return c_vision.RandomColorAdjust(contrast=(v, v))

def autocontrast_impl(level):
    v = level
    assert v == level
    return c_vision.AutoContrast()

def sharpness_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return c_vision.RandomSharpness(degrees=(v, v))

def brightness_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return c_vision.RandomColorAdjust(brightness=(v, v))

# define the Auto Augmentation policy
imagenet_policy = [
    [(posterize_impl(8), 0.4), (rotate_impl(9), 0.6)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
    [(c_vision.Equalize(), 0.8), (c_vision.Equalize(), 0.6)],
    [(posterize_impl(7), 0.6), (posterize_impl(6), 0.6)],
    [(c_vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],

    [(c_vision.Equalize(), 0.4), (rotate_impl(8), 0.8)],
    [(solarize_impl(3), 0.6), (c_vision.Equalize(), 0.6)],
    [(posterize_impl(5), 0.8), (c_vision.Equalize(), 1.0)],
    [(rotate_impl(3), 0.2), (solarize_impl(8), 0.6)],
    [(c_vision.Equalize(), 0.6), (posterize_impl(6), 0.4)],

    [(rotate_impl(8), 0.8), (color_impl(0), 0.4)],
    [(rotate_impl(9), 0.4), (c_vision.Equalize(), 0.6)],
    [(c_vision.Equalize(), 0.0), (c_vision.Equalize(), 0.8)],
    [(c_vision.Invert(), 0.6), (c_vision.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],

    [(rotate_impl(8), 0.8), (color_impl(2), 1.0)],
    [(color_impl(8), 0.8), (solarize_impl(7), 0.8)],
    [(sharpness_impl(7), 0.4), (c_vision.Invert(), 0.6)],
    [(shear_x(5), 0.6), (c_vision.Equalize(), 1.0)],
    [(color_impl(0), 0.4), (c_vision.Equalize(), 0.6)],

    [(c_vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(5), 0.6)],
    [(c_vision.Invert(), 0.6), (c_vision.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],
    [(c_vision.Equalize(), 0.8), (c_vision.Equalize(), 0.6)],
]

def split_imgs_and_labels(imgs, labels):
    """split data into labels and images"""
    ret_imgs = []
    ret_labels = []

    for i, image in enumerate(imgs):
        ret_imgs.append(image)
        ret_labels.append(labels[i])
    return np.array(ret_imgs), np.array(ret_labels)


def create_dataset(batch_size, train_data_url='', workers=8, target='Ascend', distributed=False,
                   input_size=224, color_jitter=0.5, autoaugment=False):
    """Create ImageNet training dataset"""
    if not os.path.exists(train_data_url):
        raise ValueError('Path not exists')
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distributed:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
            rank_id = 1
    if device_num == 1:
        dataset_train = ds.ImageFolderDataset(train_data_url, num_parallel_workers=workers, shuffle=True)
    else:
        dataset_train = ds.ImageFolderDataset(train_data_url, num_parallel_workers=workers, shuffle=True,
                                              num_shards=device_num, shard_id=rank_id)


    type_cast_op = c_transforms.TypeCast(mstype.int32)

    random_resize_crop_bicubic = c_vision.RandomCropDecodeResize(input_size, scale=SCALE,
                                                                 ratio=RATIO, interpolation=Inter.BICUBIC)

    random_horizontal_flip_op = c_vision.RandomHorizontalFlip(prob=0.5)
    adjust_range = (max(0, 1 - color_jitter), 1 + color_jitter)
    random_color_jitter_op = c_vision.RandomColorAdjust(brightness=adjust_range,
                                                        contrast=adjust_range,
                                                        saturation=adjust_range)
    normalize_op = c_vision.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    channel_op = c_vision.HWC2CHW()

    # assemble all the transforms
    if autoaugment:
        dataset_train = dataset_train.map(input_columns=["image"],
                                          operations=[random_resize_crop_bicubic],
                                          num_parallel_workers=workers)
        dataset_train = dataset_train.map(input_columns=["image"],
                                          operations=c_vision.RandomSelectSubpolicy(imagenet_policy),
                                          num_parallel_workers=workers)
        dataset_train = dataset_train.map(input_columns=["image"],
                                          operations=[random_horizontal_flip_op, normalize_op, channel_op],
                                          num_parallel_workers=workers)
    else:
        image_ops = [random_resize_crop_bicubic, random_horizontal_flip_op,
                     random_color_jitter_op, normalize_op, channel_op]

        dataset_train = dataset_train.map(input_columns=["image"],
                                          operations=image_ops,
                                          num_parallel_workers=workers)

    dataset_train = dataset_train.map(input_columns=["label"],
                                      operations=type_cast_op,
                                      num_parallel_workers=workers)

    # batch dealing
    ds_train = dataset_train.batch(batch_size, drop_remainder=True)

    ds_train = ds_train.repeat(1)
    return ds_train


def create_dataset_val(batch_size=128, val_data_url='', workers=8, target='Ascend', distributed=False,
                       input_size=224):
    """Create ImageNet validation dataset"""
    if not os.path.exists(val_data_url):
        raise ValueError('Path not exists')

    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distributed:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1

    if device_num == 1:
        dataset = ds.ImageFolderDataset(val_data_url, num_parallel_workers=workers)
    else:
        dataset = ds.ImageFolderDataset(val_data_url, num_parallel_workers=workers,
                                        num_shards=device_num, shard_id=rank_id)

    scale_size = None

    if isinstance(input_size, tuple):
        assert len(input_size) == 2
        if input_size[-1] == input_size[-2]:
            scale_size = int(math.floor(input_size[0] / DEFAULT_CROP_PCT))
        else:
            scale_size = tuple([int(x / DEFAULT_CROP_PCT) for x in input_size])
    else:
        scale_size = int(math.floor(input_size / DEFAULT_CROP_PCT))

    type_cast_op = c_transforms.TypeCast(mstype.int32)
    decode_op = c_vision.Decode()
    resize_op = c_vision.Resize(size=scale_size, interpolation=Inter.BICUBIC)
    center_crop = c_vision.CenterCrop(size=input_size)
    normalize_op = c_vision.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    channel_op = c_vision.HWC2CHW()

    image_ops = [decode_op, resize_op, center_crop, normalize_op, channel_op]

    dataset = dataset.map(input_columns=["label"], operations=type_cast_op,
                          num_parallel_workers=workers)
    dataset = dataset.map(input_columns=["image"], operations=image_ops,
                          num_parallel_workers=workers)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.repeat(1)
    return dataset

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
