# Copyright 2021 Huawei Technologies Co., Ltd
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
"""ubc dataloader"""
import os
import random
import numpy as np
from tqdm import tqdm
import cv2
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from src.prepare_data import prepare_data


class TripletPhotoTour:
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """

    def __init__(self,
                 root,
                 name,
                 n_triplets=5000000,
                 train=True,
                 batch_size=None,
                 load_random_triplets=False):
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size

        prepare_data(root, name)
        ckpt_name = '{}.ckpt'.format(name)
        ckpt_path = os.path.join(root, ckpt_name)
        ckpt_data = load_checkpoint(ckpt_path)
        self.data = Tensor(ckpt_data['data'], dtype=mstype.uint8).asnumpy()
        self.labels = Tensor(ckpt_data['labels']).asnumpy()
        self.matches = Tensor(ckpt_data['matches']).asnumpy()
        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels,
                                                   self.n_triplets,
                                                   self.batch_size)

    @staticmethod
    def generate_triplets(labels, num_triplets, batch_size):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels)
        n_classes = unique_labels.shape[0]
        already_idxs = set()

        for _ in tqdm(range(num_triplets)):
            if len(already_idxs) >= batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append(
                [indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return np.array(triplets)

    def __getitem__(self, index):
        if not self.train:
            m_0, m_1, m_2 = self.matches[index]
            img1 = self.data[m_0]
            img2 = self.data[m_1]

            return img1, img2, m_2

        i_0, i_1, i_2 = self.triplets[index]
        img_a, img_p, img_n = self.data[i_0], self.data[i_1], self.data[i_2]

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return len(self.triplets)
        return len(self.matches)


class TripletDataAugment:
    def __init__(self,
                 shape,
                 mean_image,
                 std_image,
                 h_flip_prob=0.5,
                 v_flip_prob=0.5,
                 rotate_prob=0.5):

        if isinstance(shape, int):
            shape = (shape, shape)

        self.shape = shape
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotate_prob = rotate_prob
        self.mean_image = mean_image
        self.std_image = std_image

    def img_flip(self, img_1, img_2, img_3=None):
        if random.random() < self.h_flip_prob:
            img_1 = cv2.flip(img_1, 1)
            img_2 = cv2.flip(img_2, 1)
            if img_3 is not None:
                img_3 = cv2.flip(img_3, 1)

        if random.random() < self.v_flip_prob:
            img_1 = cv2.flip(img_1, 0)
            img_2 = cv2.flip(img_2, 0)
            if img_3 is not None:
                img_3 = cv2.flip(img_3, 0)

        return img_1, img_2, img_3

    def img_rotate(self, img_1, img_2, img_3=None):

        if not random.random() < self.rotate_prob:
            return img_1, img_2, img_3

        degree = random.choice([90, -90])
        h, w = img_1.shape[:2]
        center = ((w - 1) * 0.5, (h - 1) * 0.5)

        rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

        img_1 = cv2.warpAffine(img_1,
                               rotation_matrix, (w, h),
                               flags=cv2.INTER_LINEAR)
        img_2 = cv2.warpAffine(img_2,
                               rotation_matrix, (w, h),
                               flags=cv2.INTER_LINEAR)
        if img_3 is not None:
            img_3 = cv2.warpAffine(img_3,
                                   rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR)

        return img_1, img_2, img_3

    def img_resize(self, img):
        return cv2.resize(img, self.shape, interpolation=cv2.INTER_LINEAR)

    def __call__(self, img_1, img_2, img_3=None):
        img_1 = self.img_resize(img_1)
        img_2 = self.img_resize(img_2)
        img_1 = (img_1 - self.mean_image) / self.std_image
        img_2 = (img_2 - self.mean_image) / self.std_image
        if img_3 is not None:
            img_3 = self.img_resize(img_3)

        img_1, img_2, img_3 = self.img_flip(img_1, img_2, img_3)
        img_1, img_2, img_3 = self.img_rotate(img_1, img_2, img_3)

        img_1 = np.float32(img_1)
        img_2 = np.float32(img_2)
        if img_3 is not None:
            img_3 = np.float32(img_3)

        cv2.normalize(img_1, img_1, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(img_2, img_2, 0, 255, cv2.NORM_MINMAX)
        if img_3 is not None:
            cv2.normalize(img_3, img_3, 0, 255, cv2.NORM_MINMAX)

        if img_3 is not None:
            return img_1[None, :, :], img_2[None, :, :], img_3[None, :, :]
        return img_1[None, :, :], img_2[None, :, :]


def create_loaders(load_random_triplets=False, args=None):
    ubc_subset = ['yosemite', 'notredame', 'liberty']

    test_dataset_names = []
    for val in ubc_subset:
        if val != args.training_set:
            test_dataset_names.append(val)

    img_size = args.imageSize
    if args.fliprot:
        train_triplet_aug_op = TripletDataAugment(img_size, args.mean_image, args.std_image, 0.5, 0, 0.5)
    else:
        train_triplet_aug_op = TripletDataAugment(img_size, args.mean_image, args.std_image, 0, 0, 0)

    test_triplet_aug_op = TripletDataAugment(img_size, args.mean_image, args.std_image, 0, 0, 0)

    train_dataset = ds.GeneratorDataset(
        TripletPhotoTour(root=args.dataroot,
                         name=args.training_set,
                         n_triplets=args.n_triplets,
                         train=True,
                         batch_size=args.batch_size,
                         load_random_triplets=load_random_triplets),
        column_names=["data_a", "data_p", "data_n"],
        shuffle=True,
        shard_id=args.rank,
        num_shards=args.group_size)

    train_dataset = train_dataset.map(
        operations=train_triplet_aug_op,
        input_columns=['data_a', 'data_p', 'data_n'],
        num_parallel_workers=8)

    train_dataset = train_dataset.batch(args.batch_size)

    eval_dataset = [{
        'name':
            name,
        'dataloader':
            ds.GeneratorDataset(
                TripletPhotoTour(root=args.dataroot,
                                 name=name,
                                 n_triplets=args.n_triplets,
                                 train=False,
                                 batch_size=args.test_batch_size,
                                 load_random_triplets=load_random_triplets),
                column_names=["data_a", "data_p", "label"],
                shuffle=False)
    } for name in test_dataset_names]

    typecast_op = C.TypeCast(mstype.int32)
    for test_dataset in eval_dataset:
        test_dataset['dataloader'] = test_dataset['dataloader'].map(
            operations=test_triplet_aug_op,
            input_columns=['data_a', 'data_p'],
            num_parallel_workers=8)

        test_dataset['dataloader'] = test_dataset['dataloader'].map(
            operations=typecast_op,
            input_columns=['label'],
            num_parallel_workers=8)

        test_dataset['dataloader'] = test_dataset['dataloader'].batch(
            args.test_batch_size)

    return train_dataset, eval_dataset


def create_evalloaders(load_random_triplets=False, config=None):
    img_size = config.imageSize
    test_triplet_aug_op = TripletDataAugment(img_size, config.mean_image, config.std_image, 0, 0, 0)

    eval_dataset = ds.GeneratorDataset(
        TripletPhotoTour(root=config.dataroot,
                         name=config.eval_set,
                         n_triplets=config.n_triplets,
                         train=False,
                         batch_size=config.test_batch_size,
                         load_random_triplets=load_random_triplets),
        column_names=["data_a", "data_p", "label"],
        shuffle=False)

    typecast_op = C.TypeCast(mstype.int32)
    eval_dataset = eval_dataset.map(
        operations=test_triplet_aug_op,
        input_columns=['data_a', 'data_p'],
        num_parallel_workers=8)

    eval_dataset = eval_dataset.map(
        operations=typecast_op,
        input_columns=['label'],
        num_parallel_workers=8)

    eval_dataset = eval_dataset.batch(
        config.test_batch_size)

    return eval_dataset
