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
"""create train or eval dataset."""

import os
import warnings
from io import BytesIO
from PIL import Image
import numpy as np

import mindspore.dataset as de
import mindspore.common.dtype as mstype
from mindspore.dataset.vision.utils import Inter
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

from src.datasets.mixup import Mixup
from src.datasets.random_erasing import RandomErasing
from src.datasets.auto_augment import rand_augment_transform

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def create_dataset(dataset_path,
                   do_train=True,
                   image_size=224,
                   interpolation="BICUBIC",
                   crop_min=0.08,
                   repeat_num=1,
                   batch_size=32,
                   num_workers=8,
                   auto_augment="rand-m9-mstd0.5-inc1",
                   mixup=0.0,
                   mixup_prob=1.0,
                   switch_prob=0.5,
                   cutmix=1.0,
                   hflip=0.5,
                   re_prop=0.25,
                   re_mode='pixel',
                   re_count=1,
                   label_smoothing=0.1,
                   num_classes=1001):
    """create_dataset"""
    device_num = int(os.getenv("RANK_SIZE", '1'))
    rank_id = int(os.getenv('RANK_ID', '0'))

    if hasattr(Inter, interpolation):
        interpolation = getattr(Inter, interpolation)
    else:
        interpolation = Inter.BICUBIC
        print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BICUBIC'))

    if do_train:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers, shuffle=True,
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

    # define map operations
    if do_train:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        assert auto_augment.startswith('rand')
        aa_params['interpolation'] = interpolation
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0), ratio=(3 / 4, 4 / 3),
                                     interpolation=interpolation),
            C.RandomHorizontalFlip(prob=hflip),
            C.ToPIL()
        ]
        trans += [rand_augment_transform(auto_augment, aa_params)]
        trans += [
            C.ToTensor(),
            C.Normalize(mean=mean, std=std, is_hwc=False),
            RandomErasing(probability=re_prop, mode=re_mode, max_count=re_count)
        ]

    else:
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        trans = [
            C.Decode(),
            C.Resize(int(256 / 224 * image_size), interpolation=interpolation),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std, is_hwc=True),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", num_parallel_workers=num_workers, operations=trans, python_multiprocessing=True)
    ds = ds.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)

    ds = ds.batch(batch_size, drop_remainder=True)

    if (mixup > 0. or cutmix > 0.) and do_train:
        mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=cutmix,
            cutmix_minmax=None, prob=mixup_prob,
            switch_prob=switch_prob,
            label_smoothing=label_smoothing,
            num_classes=num_classes)

        ds = ds.map(operations=mixup_fn, input_columns=["image", "label"],
                    num_parallel_workers=num_workers)

    ds = ds.repeat(repeat_num)
    return ds


def get_dataset(args, is_train=True):
    if args.dataset_name == "imagenet":
        if is_train:
            data = create_dataset(dataset_path=args.dataset_path,
                                  image_size=args.image_size,
                                  interpolation=args.interpolation,
                                  auto_augment=args.auto_augment,
                                  mixup=args.mixup,
                                  cutmix=args.cutmix,
                                  mixup_prob=args.mixup_prob,
                                  switch_prob=args.switch_prob,
                                  re_prop=args.re_prop,
                                  re_mode=args.re_mode,
                                  re_count=args.re_count,
                                  label_smoothing=args.label_smoothing,
                                  crop_min=args.crop_min,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        else:
            data = create_dataset(dataset_path=args.eval_path,
                                  do_train=False,
                                  image_size=args.image_size,
                                  interpolation=args.interpolation,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    elif args.dataset_name == "cifar10":
        if is_train:
            data = create_dataset(dataset_path=args.dataset_path,
                                  image_size=args.image_size,
                                  interpolation=args.interpolation,
                                  auto_augment=args.auto_augment,
                                  mixup=args.mixup,
                                  cutmix=args.cutmix,
                                  mixup_prob=args.mixup_prob,
                                  switch_prob=args.switch_prob,
                                  re_prop=args.re_prop,
                                  re_mode=args.re_mode,
                                  re_count=args.re_count,
                                  label_smoothing=args.label_smoothing,
                                  crop_min=args.crop_min,
                                  batch_size=args.batch_size,
                                  num_classes=10,
                                  num_workers=args.num_workers)
        else:
            data = create_dataset(dataset_path=args.eval_path,
                                  do_train=False,
                                  image_size=args.image_size,
                                  interpolation=args.interpolation,
                                  batch_size=args.batch_size,
                                  num_classes=10,
                                  num_workers=args.num_workers)
    else:
        raise NotImplementedError
    return data
