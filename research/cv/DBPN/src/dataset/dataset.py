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

"""return train and eval dataset"""
import math
import os
import random
import numpy as np
from PIL import Image, ImageOps
import mindspore.dataset.vision as V
from mindspore import dataset as de, context
from mindspore.context import ParallelMode
from mindspore.communication import get_rank, get_group_size


def is_image_file(filename):
    """Judge whether it is a picture."""
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def rescale_img(img_in, scale):
    """rescale_img  """
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, img_bic, args, ix=-1, iy=-1):
    """patch the image"""
    (ih, iw) = img_in.size
    scale = args.upscale_factor
    patch_size = args.patch_size
    tp = scale * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))
    return img_in, img_tar, img_bic


def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    """data augment"""
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True
    return img_in, img_tar, img_bic


class DBPNDataset():
    """
    This dataset class can load images from image folder.
    Args:
        hr_dir (str): HR Images root directory.
    Returns:
        Image path list.
    """

    def __init__(self, hr_dir, args):
        super(DBPNDataset, self).__init__()
        self.hr_dir = hr_dir
        self.list_files = sorted(os.listdir(self.hr_dir))
        self.image_filenames = [os.path.join(self.hr_dir, x) for x in self.list_files if is_image_file(x)]
        self.size = len(self.list_files)
        self.args = args

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img_path = self.image_filenames[index % self.size]
        hr = Image.open(img_path).convert('RGB')
        w, h = hr.size
        lr = hr.resize((int(w / self.args.upscale_factor), int(h / self.args.upscale_factor)), Image.BICUBIC)
        bicubic = rescale_img(lr, self.args.upscale_factor)
        lr, hr, bicubic = get_patch(lr, hr, bicubic, self.args)
        if self.args.data_augmentation:
            hr, lr, bicubic = augment(hr, lr, bicubic)
        return hr, lr, bicubic


class DistributedSampler:
    """Distributed sampler."""
    def __init__(self, dataset_size, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            print("***********Setting world_size to 1 since it is not passed in ******************")
            num_replicas = 1
        if rank is None:
            print("***********Setting rank to 0 since it is not passed in ******************")
            rank = 0
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            # np.array type. number from 0 to len(dataset_size)-1, used as index of dataset
            indices = indices.tolist()
            self.epoch += 1
            # change to list type
        else:
            indices = list(range(self.dataset_size))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def create_train_dataset(dataset, args):
    """return train dataset """
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
        rank = get_rank()
        device_num = get_group_size()
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=True)
        train_ds = de.GeneratorDataset(dataset, column_names=['target_image', 'input_image', 'bicubic_image'],
                                       sampler=distributed_sampler)
    else:
        train_ds = de.GeneratorDataset(dataset, column_names=['target_image', 'input_image', 'bicubic_image'],
                                       shuffle=True)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not args.vgg:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    trans = [
        V.ToTensor(),
    ]
    if args.isgan:
        trans.append(V.Normalize(mean=mean, std=std, is_hwc=False))
    train_ds = train_ds.map(operations=trans, input_columns=['target_image'])
    train_ds = train_ds.map(operations=trans, input_columns=['input_image'])
    train_ds = train_ds.map(operations=trans, input_columns=['bicubic_image'])
    train_ds = train_ds.batch(args.batchSize, drop_remainder=True)
    return train_ds


class DatasetVal():
    """Define val dataset"""

    def __init__(self, hr_dir, lr_dir, args):
        super(DatasetVal, self).__init__()
        self.hr_imgs = sorted(os.listdir(hr_dir))
        self.lr_imgs = sorted(os.listdir(lr_dir))
        self.hr_img_files = [os.path.join(hr_dir, x) for x in self.hr_imgs if is_image_file(x)]
        self.lr_img_files = [os.path.join(lr_dir, x) for x in self.lr_imgs if is_image_file(x)]
        self.args = args

    def __len__(self):
        return len(self.hr_img_files)

    def __getitem__(self, index):
        hr_path = self.hr_img_files[index]
        lr_path = self.lr_img_files[index]
        hr = Image.open(hr_path).convert('RGB')
        lr = Image.open(lr_path).convert('RGB')
        bicubic = rescale_img(lr, self.args.upscale_factor)
        return hr, lr, bicubic


def create_val_dataset(dataset, args):
    """Create val dataset."""
    val_ds = de.GeneratorDataset(dataset, column_names=["target_image", "input_image", "bicubic_image"], shuffle=False)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not args.vgg:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    trans = [V.ToTensor()]
    if args.isgan:
        trans.append(V.Normalize(mean=mean, std=std, is_hwc=False))
    val_ds = val_ds.map(operations=trans, input_columns=["target_image"])
    val_ds = val_ds.map(operations=trans, input_columns=["input_image"])
    val_ds = val_ds.map(operations=trans, input_columns=["bicubic_image"])
    val_ds = val_ds.batch(args.testBatchSize, drop_remainder=True)
    return val_ds
