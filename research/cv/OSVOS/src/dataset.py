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
get dataloader.
"""
import os
import math
import cv2
import numpy as np
import mindspore
import mindspore.dataset.vision.c_transforms as vision
from mindspore import Tensor
from src.utils import ScaleNRotate, RandomHorizontalFlip


class Imagelist:
    """class for loading dataset."""
    def __init__(self,
                 train=True,
                 db_root_dir='./DAVIS',
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        self.train = train
        self.db_root_dir = db_root_dir
        self.meanval = meanval
        self.seq_name = seq_name
        if self.train:
            fname = os.path.join(db_root_dir, 'train.txt')
        else:
            fname = os.path.join(db_root_dir, 'val.txt')
        if self.seq_name is None:
            with open(fname) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    seq_name = seq.strip()
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq_name)))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq_name, x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq_name)))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq_name, x), lab))
                    labels.extend(lab_path)
        else:
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]
        assert len(labels) == len(img_list)
        self.img_list = img_list
        self.labels = labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        if self.train:
            sample = {'image': img, 'gt': gt}
            sample = RandomHorizontalFlip()(sample)
            sample = ScaleNRotate()(sample)
            img = sample['image']
            gt = sample['gt']
        if not self.train and self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return Tensor(img, mindspore.float32), fname
        return img, gt

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair.
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt/np.max([gt.max(), 1e-8])
        return img, gt

class DistributedSampler:
    """
    Distributed sampler.
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset.classes)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank::self.group_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

def create_dataset(mode, data_path, batch_size, seq_name=None, num_of_workers=4, num_of_epoch=1,
                   is_distributed=False, rank=0, group_size=1, seed=0):
    """create dataset for train or test."""

    shuffle, drop_last, train_mode = None, None, None
    if mode == 'Train':
        shuffle = True
        drop_last = True
        train_mode = True

    else:
        shuffle = False
        drop_last = False
        train_mode = False
    dataset_generator = Imagelist(train_mode, data_path, seq_name=seq_name)
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset=dataset_generator, rank=rank,
                                     group_size=group_size, shuffle=shuffle, seed=seed)
    dataset = mindspore.dataset.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=shuffle,
                                                 sampler=sampler, num_parallel_workers=num_of_workers)
    dataset = dataset.map(input_columns=["image"], operations=vision.HWC2CHW(), num_parallel_workers=num_of_workers)
    dataset = dataset.batch(batch_size, num_parallel_workers=num_of_workers, drop_remainder=drop_last)
    dataset = dataset.repeat(num_of_epoch)
    return dataset
