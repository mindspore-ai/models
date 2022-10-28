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
"""create train or eval dataset."""

import os
import warnings
from io import BytesIO
from PIL import Image
import numpy as np

import mindspore as ms
import mindspore.dataset.engine as de
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision.utils import Inter

from .autoaugment import ImageNetPolicy

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ToNumpy:
    def __init__(self):
        pass

    def __call__(self, img):
        return np.asarray(img)


def create_dataset(dataset_path,
                   do_train,
                   image_size=224,
                   interpolation='BILINEAR',
                   crop_min=0.05,
                   repeat_num=1,
                   batch_size=32,
                   num_workers=12,
                   autoaugment=False,
                   mixup=0.0,
                   num_classes=1001):
    """create_dataset"""

    if hasattr(Inter, interpolation):
        interpolation = getattr(Inter, interpolation)
    else:
        interpolation = Inter.BILINEAR
        print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))

    device_num = int(os.getenv("RANK_SIZE", '1'))
    rank_id = int(os.getenv('RANK_ID', '0'))

    if do_train:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=1, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)
    else:
        batch_per_step = batch_size * device_num
        print("eval batch per step: {}".format(batch_per_step))
        if batch_per_step < 50000:
            if 50000 % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (50000 % batch_per_step)
        else:
            num_padded = batch_per_step - 50000
        print("eval dataset num_padded: {}".format(num_padded))

        if num_padded != 0:
            # padded_with_decode
            white_io = BytesIO()
            Image.new('RGB', (image_size, image_size), (255, 255, 255)).save(white_io, 'JPEG')
            padded_sample = {
                'image': np.array(bytearray(white_io.getvalue()), dtype='uint8'),
                'label': np.array(-1, np.int32)
            }
            sample = [padded_sample for x in range(num_padded)]
            ds_pad = de.PaddedDataset(sample)
            ds_imagefolder = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers)
            ds = ds_pad + ds_imagefolder
            distribute_sampler = de.DistributedSampler(num_shards=device_num, shard_id=rank_id,
                                                       shuffle=False, num_samples=None)
            ds.use_sampler(distribute_sampler)
        else:
            ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers,
                                       shuffle=False, num_shards=device_num, shard_id=rank_id)
            print("eval dataset size: {}".format(ds.get_dataset_size()))

    # Computed from random subset of ImageNet training images
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    ds = ds.repeat(repeat_num)

    # define map operations
    if do_train:
        c_trans = [
            vision.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0),
                                          ratio=(0.75, 1.333), interpolation=interpolation),
            vision.RandomHorizontalFlip(prob=0.5),
        ]
        ds = ds.map(input_columns="image", num_parallel_workers=num_workers, operations=c_trans)
        if autoaugment:
            py_trans = [
                vision.ToPIL(),
                ImageNetPolicy(),
                ToNumpy(),
            ]
            ds = ds.map(input_columns="image", num_parallel_workers=8, operations=py_trans, python_multiprocessing=True)
        extra_trans = [
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW(),
        ]
        ds = ds.map(input_columns="image", num_parallel_workers=2, operations=extra_trans)
    else:
        resize = int(int(image_size / 0.875 / 16 + 0.5) * 16)
        print('eval, resize:{}'.format(resize))
        c_trans = [
            vision.Decode(),
            vision.Resize(resize, interpolation=interpolation),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
        ds = ds.map(input_columns="image", num_parallel_workers=num_workers, operations=c_trans)

    type_cast_op = transforms.TypeCast(ms.int32)
    ds = ds.map(input_columns="label", num_parallel_workers=1, operations=type_cast_op)

    if do_train and mixup > 0:
        one_hot_encode = transforms.OneHot(num_classes)
        ds = ds.map(operations=one_hot_encode, input_columns=["label"], num_parallel_workers=1)

    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_workers=1)

    if do_train and mixup > 0:
        trans_mixup = vision.MixUpBatch(alpha=mixup)
        ds = ds.map(input_columns=["image", "label"], num_parallel_workers=num_workers, operations=trans_mixup)

    return ds


def get_dataset(dataset_name, do_train, dataset_path, args):
    if dataset_name == "imagenet":
        if do_train:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=True,
                                  image_size=args.train_image_size,
                                  interpolation=args.interpolation,
                                  autoaugment=args.autoaugment,
                                  mixup=args.mixup,
                                  crop_min=args.crop_min,
                                  batch_size=args.batch_size,
                                  num_workers=args.train_num_workers)
        else:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=False,
                                  image_size=args.eval_image_size,
                                  interpolation=args.interpolation,
                                  batch_size=args.eval_batch_size,
                                  num_workers=args.eval_num_workers)
    elif dataset_name == "cifar10":
        if do_train:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=True,
                                  image_size=args.train_image_size,
                                  interpolation=args.interpolation,
                                  autoaugment=args.autoaugment,
                                  mixup=args.mixup,
                                  crop_min=args.crop_min,
                                  batch_size=args.batch_size,
                                  num_classes=10,
                                  num_workers=args.train_num_workers)
        else:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=False,
                                  image_size=args.eval_image_size,
                                  interpolation=args.interpolation,
                                  batch_size=args.eval_batch_size,
                                  num_classes=10,
                                  num_workers=args.eval_num_workers)
    else:
        raise NotImplementedError
    return data
