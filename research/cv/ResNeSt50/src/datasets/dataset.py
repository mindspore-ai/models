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
"""Dataset"""
import os

import mindspore.dataset as dataset
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as V_C
from mindspore.common import dtype as mstype

from src.datasets.autoaug import RandAugment

def ImageNet(root, mode,
             img_size, crop_size,
             rank, group_size, epoch, batch_size,
             transform=None, target_transform=None,
             shuffle=None, sampler=None,
             class_indexing=None, drop_remainder=True,
             num_parallel_workers=None, **kwargs):
    """ImageNet dataset"""
    split = "train" if mode == "train" else "val"
    data_path = os.path.join(root, split)

    if transform is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_img = []
        if mode == "train":
            transform_img += [
                V_C.Decode(),
                V_C.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                V_C.RandomHorizontalFlip(prob=0.5),
                V_C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                V_C.ToPIL(),
                RandAugment(2, 12, True, True),
                V_C.ToTensor(),
                V_C.Normalize(mean=mean, std=std, is_hwc=False)]
        else:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            transform_img = [
                V_C.Decode(),
                V_C.Resize((320, 320)),
                V_C.CenterCrop(256),
                V_C.Normalize(mean=mean, std=std, is_hwc=True),
                V_C.HWC2CHW()]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform

    imagenet = dataset.ImageFolderDataset(data_path, num_parallel_workers=num_parallel_workers,
                                          shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                          num_shards=group_size, shard_id=rank)
    imagenet = imagenet.map(operations=transform_img, input_columns="image",
                            num_parallel_workers=num_parallel_workers)
    imagenet = imagenet.map(operations=transform_label, input_columns="label",
                            num_parallel_workers=num_parallel_workers)
    columns_to_project = ["image", "label"]
    imagenet = imagenet.project(columns=columns_to_project)

    imagenet = imagenet.batch(batch_size, drop_remainder=drop_remainder)

    return imagenet
