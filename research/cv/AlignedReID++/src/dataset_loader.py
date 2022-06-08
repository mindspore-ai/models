"""get the dataset"""
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
from __future__ import print_function, absolute_import

import random
import math
import os.path as osp
from PIL import Image

from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms.transforms import Compose
import mindspore.dataset.vision as vision

from .import data_manager
from .import samplers

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))

    return img

class Random2DTranslation():
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

class RandomErasing():
    """data augmentation random erase"""

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        image_c, image_h, image_w = img.shape
        area = image_h * image_w

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < image_w and h < image_h:
            x1 = random.randint(0, image_h - h)
            y1 = random.randint(0, image_w - w)
            if image_c == 3:
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
            else:
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
            return img

        return img

class ImageDataset():
    """Image Person ReID Dataset"""
    def __init__(self, dataset, class_num, rank_size, num_instances):
        self.dataset = dataset
        self.rank_size = rank_size
        self.num_instances = num_instances
        self.class_num = class_num

        self.pids_per_npu = self.class_num//self.rank_size
        self.samples_per_npu = self.pids_per_npu*self.num_instances

    def __len__(self):
        return self.samples_per_npu

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = read_image(img_path)
        return img, pid, camid


class ImageDatasetTest():
    """Image Person ReID Dataset"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = read_image(img_path)
        return img, pid, camid

def decode(img):
    """convert the img"""
    img = Image.fromarray(img)
    return img

def create_train_dataset(real_path, args, rank_id, rank_size):
    """create the train dataset"""
    dataset = data_manager.init_img_dataset(root=real_path, name=args.dataset)
    transform_train = [
        decode,
        Random2DTranslation(args.height, args.width),
        vision.RandomHorizontalFlip(0.5),
        vision.ToTensor(),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
        RandomErasing()
    ]
    transform_train = Compose(transform_train)
    source_dataset = ImageDataset(dataset.train, dataset.num_train_pids, rank_size, args.num_instances)
    Sampler = samplers.RandomIdentitySampler(dataset.train, rank_id, rank_size, args.num_instances)

    trainloader = GeneratorDataset(
        source=source_dataset,
        column_names=["img", "pid", "camid"],
        sampler=Sampler,
        num_parallel_workers=args.workers
    )

    trainloader = trainloader.map(input_columns="img", operations=transform_train)
    trainloader = trainloader.batch(batch_size=32, drop_remainder=True)

    return trainloader, dataset.num_train_pids

def create_test_dataset(real_path, args):
    """create the test dataset"""
    dataset = data_manager.init_img_dataset(root=real_path, name=args.dataset)

    transform_test = [
        decode,
        vision.Resize((args.height, args.width)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
    ]
    transform_test = Compose(transform_test)

    queryloader = GeneratorDataset(
        source=ImageDatasetTest(dataset.query),
        column_names=["img", "pid", "camid"],
        shuffle=False,
        num_parallel_workers=args.workers
    )

    galleryloader = GeneratorDataset(
        source=ImageDatasetTest(dataset.gallery),
        column_names=["img", "pid", "camid"],
        shuffle=False,
        num_parallel_workers=args.workers
    )

    queryloader = queryloader.map(input_columns="img", operations=transform_test)
    galleryloader = galleryloader.map(input_columns="img", operations=transform_test)
    queryloader = queryloader.batch(batch_size=32, drop_remainder=True)
    galleryloader = galleryloader.batch(batch_size=32, drop_remainder=True)

    return queryloader, galleryloader, dataset.num_train_pids

