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


import os
import sys
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.dataset.vision.c_transforms as C
from PIL import Image
sys.path.append("..")


class TrainData:
    """
    dataloader for pageNet
    """

    # image_root="D:\\Page-Net-pytorch\\image", gt_root="D:\\Page-Net-pytorch\\groundTruth"
    def __init__(self, image_root, gt_root, edge_root, img_size, augmentations):
        self.img_size = img_size
        self.augmentations = augmentations
        self.images = [image_root + "/" + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + "/" + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + "/" + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.size = len(self.images)

        print('no augmentation')
        self.img_transform = transforms.c_transforms.Compose([
            C.Resize((self.img_size, self.img_size)),
            vision.py_transforms.ToTensor()])
        self.gt_transform = transforms.c_transforms.Compose([
            C.Resize((self.img_size, self.img_size)),
            vision.py_transforms.ToTensor()])
        self.edge_transform = transforms.c_transforms.Compose([
            C.Resize((self.img_size, self.img_size)),
            vision.py_transforms.ToTensor()])

    def __getitem__(self, index):

        img = Image.open(self.images[index], 'r').convert('RGB')

        gt = Image.open(self.gts[index], 'r').convert('1')

        edge = Image.open(self.edges[index], 'r').convert('1')

        if self.img_transform is not None:
            img = np.array(img, dtype=np.float32)
            img -= np.array((104.00699, 116.66877, 122.67892))
            img = self.img_transform(img)
            img = img * 255

        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
            gt = gt * 255

        if self.edge_transform is not None:
            edge = self.edge_transform(edge)
            edge = edge * 255

        return img, gt, edge


    def __len__(self):
        return self.size

class TestData:
    """
    dataloader for pageNet
    """

    # image_root="D:\\Page-Net-pytorch\\image", gt_root="D:\\Page-Net-pytorch\\groundTruth"
    def __init__(self, image_root, gt_root, img_size, augmentations):
        self.img_size = img_size
        self.augmentations = augmentations
        self.images = [image_root + "/" + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + "/" + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)



        self.img_transform = transforms.c_transforms.Compose([
            C.Resize((self.img_size, self.img_size)),
            vision.py_transforms.ToTensor()])
        self.gt_transform = transforms.c_transforms.Compose([
            C.Resize((self.img_size, self.img_size)),
            vision.py_transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        img = Image.open(self.images[index], 'r').convert('RGB')

        gt = Image.open(self.gts[index], 'r').convert('1')



        if self.img_transform is not None:
            img = np.array(img, dtype=np.float32)
            img -= np.array((104.00699, 116.66877, 122.67892))
            img = self.img_transform(img)
            img = img * 255

        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
            gt = gt * 255

        return img, gt

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.img_size or w < self.img_size:
            h = max(h, self.img_size)
            w = max(w, self.img_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        return img, gt

    def __len__(self):
        return self.size


def get_train_loader(image_root, gt_root, edge_root, batchsize, trainsize,
                     device_num=1, rank_id=0, shuffle=True,
                     num_parallel_workers=1,
                     augmentation=False):
    dataset_generator = TrainData(image_root, gt_root, edge_root, trainsize, augmentation)
    dataset = ds.GeneratorDataset(dataset_generator, ["imgs", "gts", "edges"], shuffle=shuffle,
                                  num_parallel_workers=num_parallel_workers,
                                  num_shards=device_num, shard_id=rank_id)

    data_loader = dataset.batch(batch_size=batchsize)

    return data_loader


def get_test_loader(image_root, gt_root, batchsize, testsize, augmentation=False):
    dataset_generator = TestData(image_root, gt_root, testsize, augmentation)
    dataset = ds.GeneratorDataset(dataset_generator, ["imgs", "gts"])

    data_loader = dataset.batch(batch_size=batchsize)
    return data_loader


if __name__ == '__main__':
    train_loader = get_train_loader(train_img_path, train_gt_path, train_edge_path, batchsize=1, trainsize=train_size)

    test_loader = get_test_loader(test_img_path, test_gt_path, batchsize=1, testsize=train_size)
    for data in train_loader:
        imgs, targets = data
        print(targets.shape)
