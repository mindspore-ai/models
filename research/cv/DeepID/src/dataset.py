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
"""Data Processing for DeepID"""
import os
import random
import multiprocessing
import csv
import numpy as np
from PIL import Image

import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as data_trans
import mindspore.dataset as de



def is_image_file(filename):
    """Judge whether it is an image"""
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir_path, max_dataset_size=float("inf")):
    """Return image list in dir"""
    images = []
    assert os.path.isdir(dir_path), "%s is not a valid directory" % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class Youtube:
    """
    This dataset class helps load celebA dataset.
    """
    def __init__(self, image_dir, mode, num_class=1283, transform=None):
        """Initialize and preprocess the CelebA dataset."""
        self.img_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.num_images = 0
        self.dataset = []
        self.preprocess()
        self.num_class = num_class

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        if self.mode == 'train':
            file = os.path.join(self.img_dir, 'train_set.csv')
        elif self.mode == 'valid':
            file = os.path.join(self.img_dir, 'valid_set.csv')
        elif self.mode == 'test':
            file = os.path.join(self.img_dir, 'test_set.csv')
        else:
            file = None
            print('Error mode!')
        lines = csv.reader(open(file))
        #lines.__next__()
        lines = list(lines)
        random.seed(1234)
        random.shuffle(lines)
        self.dataset = lines
        self.num_images = len(self.dataset)

        print('Finished preprocessing the Youtube dataset...')

    def __getitem__(self, idx):
        """Return one image and its corresponding attribute label."""
        dataset = self.dataset
        root_dir = os.path.join(self.img_dir, 'crop_images_DB')
        if self.mode in ['train', 'valid']:
            filename, label = dataset[idx][0].split()

            image = np.asarray(Image.open(os.path.join(root_dir, filename)))
            label = int(label)
            image = np.squeeze(self.transform(image))

        else:
            test_img1, test_img2, test_label = dataset[idx][0].split()
            test_img1 = np.asarray(Image.open(os.path.join(root_dir, test_img1)))
            test_img2 = np.asarray(Image.open(os.path.join(root_dir, test_img2)))
            test_img1 = np.squeeze(self.transform(test_img1))
            test_img2 = np.squeeze(self.transform(test_img2))
            test_label = int(test_label)

            return test_img1, test_img2, test_label

        return image, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(data_root, mode='train'):
    """Build and return a data loader."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = [vision.ToPIL()]
    if mode == 'train':
        transform.append(vision.RandomHorizontalFlip())
    transform.append(vision.ToTensor())
    transform.append(vision.Normalize(mean=mean, std=std, is_hwc=False))
    transform = data_trans.Compose(transform)

    dataset = Youtube(data_root, mode, transform=transform)

    return dataset


def dataloader(img_path, epoch, mode='train',
               batch_size=1, device_num=1, rank=0, shuffle=True):
    """Get dataloader"""

    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)

    dataset_loader = get_loader(img_path, mode=mode)
    length_dataset = len(dataset_loader)

    distributed_sampler = de.DistributedSampler(device_num, rank, shuffle=shuffle)
    if mode in ['train', 'valid']:
        dataset_column_names = ["image", "label"]
    else:
        dataset_column_names = ["image1", "image2", "label"]

    if device_num == 1:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names)
        ds = ds.batch(batch_size, num_parallel_workers=min(32, num_parallel_workers), drop_remainder=False)

    else:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names, sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=False)

    if mode == 'train':
        ds = ds.repeat(epoch)

    return ds, length_dataset
