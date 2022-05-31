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
Dataloader and preprocessing
"""
import os
import os.path
import random
import math

import numpy as np
from PIL import Image
import mindspore.dataset.vision as vision
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size


def TrainDataLoader(img_size, data_path, dataset, batch_size, distributed):
    """ DataLoader """
    train_transform = [
        vision.ToPIL(),
        vision.RandomHorizontalFlip(),
        vision.Resize((img_size + 30, img_size + 30)),
        vision.RandomCrop(img_size),
        vision.ToTensor(),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
    ]
    test_transform = [
        vision.ToPIL(),
        vision.Resize((img_size, img_size)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
    ]
    rank_size = 1
    if distributed:
        rank_id = get_rank()
        rank_size = get_group_size()
        train_generator = GetDatasetGenerator(os.path.join(data_path, dataset), 'train')
        train_data = ds.GeneratorDataset(train_generator, ["image_A", "image_B"], shuffle=True, num_parallel_workers=12,
                                         num_shards=rank_size, shard_id=rank_id)

        test_generator = GetDatasetGenerator(os.path.join(data_path, dataset), 'test')
        test_data = ds.GeneratorDataset(test_generator, ["image_A", "image_B"], shuffle=False, num_parallel_workers=12,
                                        num_shards=rank_size, shard_id=rank_id)

    else:
        train_generator = GetDatasetGenerator(os.path.join(data_path, dataset), 'train')
        train_data = ds.GeneratorDataset(train_generator, ["image_A", "image_B"], shuffle=True, num_parallel_workers=12)
        test_generator = GetDatasetGenerator(os.path.join(data_path, dataset), 'test')
        test_data = ds.GeneratorDataset(test_generator, ["image_A", "image_B"], shuffle=False, num_parallel_workers=12)

    train_data = train_data.map(operations=train_transform, input_columns=["image_A"])
    train_data = train_data.map(operations=train_transform, input_columns=["image_B"])
    train_data = train_data.batch(batch_size=batch_size).repeat(3)
    train_num = train_data.get_dataset_size()

    test_data = test_data.map(operations=test_transform, input_columns=["image_A"])
    test_data = test_data.map(operations=test_transform, input_columns=["image_B"])
    test_data = test_data.batch(batch_size=1).repeat()

    return train_data, test_data, train_num


def TestDataLoader(img_size, data_path, dataset):
    """ DataLoader """
    test_transform = [
        vision.ToPIL(),
        vision.Resize((img_size, img_size)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
    ]
    testA_generator = GetDatasetGenerator(os.path.join(data_path, dataset), 'test')
    testA = ds.GeneratorDataset(testA_generator, ["image_A", "image_B"], shuffle=False, num_parallel_workers=12)

    testA = testA.map(operations=test_transform, input_columns=["image_A"])
    testA = testA.map(operations=test_transform, input_columns=["image_B"])
    testA_loader = testA.batch(batch_size=1).repeat(1)
    return testA_loader


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(in_dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(in_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class GetDatasetGenerator:
    """Dataset"""

    def __init__(self, root, phase, loader=default_loader, extensions=None, transform=None, target_transform=None):
        if not extensions:
            extensions = IMG_EXTENSIONS
        self.root_A = os.path.join(root, phase + 'A')
        self.root_B = os.path.join(root, phase + 'B')
        samlpe_A = make_dataset(self.root_A, extensions)
        samlpe_B = make_dataset(self.root_B, extensions)
        if not samlpe_A:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
        self.phase = phase
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samlpe_A = samlpe_A
        self.samlpe_B = samlpe_B
        self.A_size = len(self.samlpe_A)
        self.B_size = len(self.samlpe_B)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        index_B = index % self.B_size
        if self.phase == 'train':
            if index % max(self.A_size, self.B_size) == 0:
                random.shuffle(self.samlpe_A)
                index_B = random.randint(0, self.B_size - 1)
        path_A, _ = self.samlpe_A[index % self.A_size]
        path_B, _ = self.samlpe_B[index_B]
        # sample = self.loader(path)
        sample_A = np.array(Image.open(path_A).convert('RGB'))
        sample_B = np.array(Image.open(path_B).convert('RGB'))

        return sample_A, sample_B

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DistributedSampler:
    """Distributed sampler."""

    def __init__(self,
                 dataset_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        if num_replicas is None:
            print(
                "***********Setting world_size to 1 since it is not passed in ******************"
            )
            num_replicas = 1
        if rank is None:
            print(
                "***********Setting rank to 0 since it is not passed in ******************"
            )
            rank = 0
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(
                self.dataset_size)
            # np.array type. number from 0 to len(dataset_size)-1, used as index of dataset
            indices = indices.tolist()
            self.epoch += 1
            # change to list type
        else:
            indices = list(range(self.dataset_size))

        # add extra samlpe_A to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
