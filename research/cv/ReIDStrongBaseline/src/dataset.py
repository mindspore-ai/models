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
""" ReID dataset processing """

import math
import random

import mindspore.dataset as ds
import mindspore.dataset.vision as C
import numpy as np
from PIL import Image

from src.datasets import init_dataset
from src.sampler import ReIDDistributedSampler


class ImageDataset:
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        """ Dataset len """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Get image and label by index """
        img_path, pid, _ = self.dataset[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img)

        return img, pid


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.486, 0.459, 0.408)):
    """ Random erasing augmentation

    Args:
        img: input image
        probability: augmentation probability
        sl: min erasing area
        sh: max erasing area
        r1: erasing ratio
        mean: erasing color
    Returns:
        augmented image
    """
    if random.uniform(0, 1) > probability:
        return img

    ch, height, width = img.shape

    for _ in range(100):
        area = height * width

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < width and h < height:
            x1 = random.randint(0, height - h)
            y1 = random.randint(0, width - w)
            if ch == 3:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = mean[2]
            else:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            return img

    return img


def create_dataset(
        image_folder,
        ims_per_id=4,
        ids_per_batch=32,
        batch_size=None,
        rank=0,
        group_size=1,
        resize_h_w=(384, 128),
        padding=10,
        num_parallel_workers=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        data_part='train',
        dataset_name='market1501',
):
    """ Crate dataloader for ReID

    Args:
        image_folder: path to image folder
        ims_per_id: number of ids in batch
        ids_per_batch: number of imager per id
        batch_size: batch size (if None then batch_size=ims_per_id*ids_per_batch)
        rank: process id
        group_size: device number
        resize_h_w: height and width of image
        padding: size of augmentation padding
        num_parallel_workers: number of parallel workers
        mean: image mean value for normalization
        std: image std value for normalization
        data_part: part of data: train|test|query
        dataset_name: name of dataset: market1501|dukemtmc

    Returns:
        if train data_part:
            dataset
        else:
            dataset, camera_ids, person_ids
    """
    mean = [m * 255 for m in mean]
    std = [s * 255 for s in std]

    if batch_size is None:
        batch_size = ids_per_batch * ims_per_id

    full_dataset = init_dataset(dataset_name, image_folder)

    if data_part == 'train':
        subset = full_dataset.train
    elif data_part == 'query':
        subset = full_dataset.query
    elif data_part == 'gallery':
        subset = full_dataset.gallery
    elif data_part == 'validation':
        subset = full_dataset.query + full_dataset.gallery
    else:
        raise ValueError(f'Unknown data_part {data_part}')

    reid_dataset = ImageDataset(subset)

    sampler, shuffle = None, None

    _, pids, camids = list(zip(*subset))

    if data_part == 'train':

        sampler = ReIDDistributedSampler(
            subset,
            batch_id=ids_per_batch,
            batch_image=ims_per_id,
            rank=rank,
            group_size=group_size,
        )

        transforms_list = [
            C.Resize(resize_h_w),
            C.RandomHorizontalFlip(),
            C.Pad(padding),
            C.RandomCrop(resize_h_w),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]
    else:
        shuffle = False

        transforms_list = [
            C.Resize(resize_h_w),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]

    dataset = ds.GeneratorDataset(
        source=reid_dataset,
        column_names=['image', 'label'],
        sampler=sampler,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
    )

    if data_part == 'train':
        dataset = dataset.map(
            operations=random_erasing,
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if data_part == 'train':
        return dataset

    return dataset, camids, pids
