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
import mindspore.dataset.vision as vision

def data_transforms(args):
    """get transform of dataset"""
    assert args.data_transforms in [
        'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']
    if args.data_transforms == 'imagenet1k_inception':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        crop_scale = 0.08
        jitter_param = 0.4
    elif args.data_transforms == 'imagenet1k_basic':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_scale = 0.08
        jitter_param = 0.4
    elif args.data_transforms == 'imagenet1k_mobile':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_scale = 0.25
        jitter_param = 0.4
    train_transforms = [vision.RandomCropDecodeResize(224, scale=(crop_scale, 1.0)),
                        vision.RandomColorAdjust(brightness=jitter_param,
                                                 contrast=jitter_param,
                                                 saturation=jitter_param),
                        vision.RandomHorizontalFlip(),
                        vision.HWC2CHW(),
                        ]
    val_transforms = [vision.Decode(True),
                      vision.Resize(256),
                      vision.CenterCrop(224),
                      vision.ToTensor(),
                      vision.Normalize(mean=mean, std=std, is_hwc=False)
                      ]
    return train_transforms, val_transforms

def create_dataset(train_transforms, val_transforms, args, dataset_path, device_num=1, rank_id=0):
    """get dataset for classification"""
    ds.config.set_prefetch_size(64)
    if not args.test_only:
        train_set = ds.ImageFolderDataset(os.path.join(dataset_path, 'train'),
                                          shuffle=True,
                                          num_parallel_workers=8,
                                          num_shards=device_num,
                                          shard_id=rank_id)
        train_set = train_set.map(operations=train_transforms,
                                  input_columns=["image"],
                                  num_parallel_workers=8,
                                  python_multiprocessing=False)
        train_set = train_set.batch(args.batch_size,
                                    num_parallel_workers=8,
                                    drop_remainder=True)
    else:
        train_set = None
    val_set = ds.ImageFolderDataset(os.path.join(dataset_path, 'val'),
                                    shuffle=True,
                                    num_parallel_workers=8,
                                    num_shards=device_num,
                                    shard_id=rank_id)
    val_set = val_set.map(operations=val_transforms,
                          input_columns=["image"],
                          num_parallel_workers=8,
                          python_multiprocessing=False)
    val_set = val_set.batch(args.batch_size,
                            num_parallel_workers=8,
                            drop_remainder=True)
    return train_set, val_set
