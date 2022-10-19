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
"""Dataset"""

import os
import sys
import json
import random
import logging

import numpy as np
from PIL import Image
from PIL import ImageFile
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as data_trans
import mindspore.ops
import mindspore.dataset as de

from .dataAugment import RandAugmentMC

ImageFile.LOAD_TRUNCATED_IMAGES = True

# used for network testing
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def make_dataset(annotation=None, unlabel_label=4):
    images = []
    annotation = json.load(open(annotation, 'r'))
    labeled_dataset = annotation['labeled_samples']
    unlabeled_dataset = annotation['unlabeled_samples']
    random.shuffle(labeled_dataset)
    random.shuffle(unlabeled_dataset)

    extend_size = len(unlabeled_dataset) / (len(labeled_dataset) * unlabel_label)
    if extend_size < 1:
        extend_num = int((1-extend_size) * len(unlabeled_dataset)) + 1
        unlabeled_dataset += unlabeled_dataset[0:extend_num]
    elif extend_size > 1:
        extend_num = int(extend_size * len(labeled_dataset)) + 1
        need_num = extend_num - len(labeled_dataset)
        labeled_dataset += labeled_dataset[0:need_num]
    elif extend_size == 1:
        pass

    if len(unlabeled_dataset) % unlabel_label != 0:
        num_need = unlabel_label - (len(unlabeled_dataset) % unlabel_label)
        unlabeled_dataset += unlabeled_dataset[0:num_need]

    if int(len(unlabeled_dataset) / unlabel_label) > len(labeled_dataset):
        diff = int(len(unlabeled_dataset) / unlabel_label) - len(labeled_dataset)
        labeled_dataset += labeled_dataset[0:diff]
    elif int(len(unlabeled_dataset) / unlabel_label) < len(labeled_dataset):
        diff = len(labeled_dataset) - int(len(unlabeled_dataset) / unlabel_label)
        unlabeled_dataset = unlabeled_dataset[0:-diff * unlabel_label]

    elif int(len(unlabeled_dataset) / unlabel_label) == len(labeled_dataset):
        pass

    for i in range(0, len(unlabeled_dataset), unlabel_label):
        item = []
        index_label = int(i / unlabel_label)
        item.extend([labeled_dataset[index_label]])
        item.extend(unlabeled_dataset[i:i + unlabel_label])
        images.append(item)
    return images


class CoMatchDatasetImageNet:
    """CoMatch Dataset for train."""
    def __init__(self, args):
        self.args = args
        self.unlabel_label = args.unlabel_label
        samples = make_dataset(self.args.annotation, self.unlabel_label)

        if not samples:
            raise RuntimeError("Found 0 sample in samples")

        self.samples = samples
        logging.info("sample len: %d", len(self.samples))

        self.random_resize_crop = vision.RandomResizedCrop(224, scale=(0.2, 1.))
        self.random_horizontal_flip = vision.RandomHorizontalFlip()
        self.to_tensor = vision.ToTensor()
        self.normalize = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        self.random_apply = data_trans.RandomApply([vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)], prob=0.8)
        self.random_grayscale = vision.RandomGrayscale(prob=0.2)

        self.unlable_randomaugmentMC = RandAugmentMC(int(args.unlabel_randomaug_count),
                                                     int(args.unlabel_randomaug_intensity))
        self.label_randomaugmentMC = RandAugmentMC(int(args.label_randomaug_count),
                                                   int(args.label_randomaug_intensity))

        self.concat = mindspore.ops.Concat(axis=0)
        self.concat = np.concatenate
        self.expand = mindspore.ops.ExpandDims()

    def __getitem__(self, index):
        """
       Args:
           index (int): Index
       Returns:
           tuple: (sample, target, path)
           sample : [label,[unlabel1_weak, unlabel2_strong0, unlabel2_strong1] ... ]
           target : [label_target, unlabel1_target, ...]
           path : [label_path, unlabel1_path, ...] //for debug//
       """
        target = []
        path = []

        #tmp for unlabel augment
        unlabel_sample_weak = []
        unlabel_sample_strong0 = []
        unlabel_sample_strong1 = []

        label_path, label_target = self.samples[index][0]
        label_sample = Image.open(label_path).convert('RGB')

        #label data transform
        label_sample_crop = self.random_resize_crop(label_sample)
        label_sample_flip = self.random_horizontal_flip(label_sample_crop)
        if self.args.label_aug:
            label_sample_random_aug = self.label_randomaugmentMC(label_sample_flip)
            label_sample_tensor = self.to_tensor(label_sample_random_aug)
            label_sample_normalize = self.normalize(label_sample_tensor)
        else:
            label_sample_tensor = self.to_tensor(label_sample_flip)
            label_sample_normalize = self.normalize(label_sample_tensor)

        target.append(label_target)
        path.append(label_path)

        ##unlabel data transform img1, img2...img4
        for i in range(self.args.unlabel_label):
            unlabel_path, unlabel_targe = self.samples[index][i+1]
            target.append(unlabel_targe)
            path.append(unlabel_path)

            unlabel_sample = Image.open(unlabel_path).convert('RGB')

            #firstly weak augment
            unlabel_sample_weak_crop = self.random_resize_crop(unlabel_sample)
            unlabel_sample_weak_flip = self.random_horizontal_flip(unlabel_sample_weak_crop)
            unlabel_sample_weak_tensor = self.to_tensor(unlabel_sample_weak_flip)
            unlabel_sample_weak_normalize = self.normalize(unlabel_sample_weak_tensor)

            #secondly strong0 augment
            unlabel_sample_strong0_crop = self.random_resize_crop(unlabel_sample)
            unlabel_sample_strong0_flip = self.random_horizontal_flip(unlabel_sample_strong0_crop)
            if self.args.unlabel_aug:
                unlabel_sample_strong0_random_aug = self.label_randomaugmentMC(unlabel_sample_strong0_flip)
                unlabel_sample_strong0_tensor = self.to_tensor(unlabel_sample_strong0_random_aug)
                unlabel_sample_strong0_normalize = self.normalize(unlabel_sample_strong0_tensor)
            else:
                unlabel_sample_strong0_tensor = self.to_tensor(unlabel_sample_strong0_flip)
                unlabel_sample_strong0_normalize = self.normalize(unlabel_sample_strong0_tensor)

            #thirdly strong1 augment
            unlabel_sample_strong1_crop = self.random_resize_crop(unlabel_sample)
            unlabel_sample_strong1_flip = self.random_horizontal_flip(unlabel_sample_strong1_crop)
            if self.args.unlabel_aug:
                unlabel_sample_strong1_random_aug = self.label_randomaugmentMC(unlabel_sample_strong1_flip)
                unlabel_sample_strong1_tensor = self.to_tensor(unlabel_sample_strong1_random_aug)
                unlabel_sample_strong1_normalize = self.normalize(unlabel_sample_strong1_tensor)
            else:
                unlabel_sample_strong1_tensor = self.to_tensor(unlabel_sample_strong1_flip)
                unlabel_sample_strong1_normalize = self.normalize(unlabel_sample_strong1_tensor)

            unlabel_sample_weak.append(unlabel_sample_weak_normalize)
            unlabel_sample_strong0.append(unlabel_sample_strong0_normalize)
            unlabel_sample_strong1.append(unlabel_sample_strong1_normalize)
        return label_sample_normalize, unlabel_sample_weak, unlabel_sample_strong0, unlabel_sample_strong1, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _find_classes(folder):
        """
        Finds the class folders in a dataset.

        Args:
            folder (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (folder), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(folder) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def concate(unlabel_weak, unlabel_strong0, unlabel_strong1):
    batch_size = unlabel_weak.shape[0]
    if batch_size == 1:
        output = unlabel_weak[0], unlabel_strong0[0], unlabel_strong1[0]
    else:
        unlabel_weak_ = unlabel_weak[0]
        unlabel_strong0_ = unlabel_strong0[0]  # [batch, 4, 3 224, 224] ==>[batch*4, ]
        unlabel_strong1_ = unlabel_strong1[0]
        for bts in range(1, batch_size):  #
            unlabel_weak_ = np.concatenate((unlabel_weak_, unlabel_weak[bts]), axis=0)
            unlabel_strong0_ = np.concatenate((unlabel_strong0_, unlabel_strong0[bts]), axis=0)
            unlabel_strong1_ = np.concatenate((unlabel_strong1_, unlabel_strong1[bts]), axis=0)

        output = unlabel_weak_, unlabel_strong0_, unlabel_strong1_
    return output


# used for network training
def create_comatch_dataset(args):
    """Create dataset for CoMatch."""
    comatch_dataset = CoMatchDatasetImageNet(args)
    ds = de.GeneratorDataset(comatch_dataset,
                             column_names=["label", "unlabel_weak", "unlabel_strong0", "unlabel_strong1", "target"],
                             shuffle=True, num_parallel_workers=4,
                             shard_id=args.rank, num_shards=args.device_num)

    ds = ds.batch(args.batch_size, num_parallel_workers=4, drop_remainder=True)
    ds = ds.map(num_parallel_workers=8, operations=[concate],
                input_columns=["unlabel_weak", "unlabel_strong0", "unlabel_strong1"])

    return ds, len(comatch_dataset)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset_test(folder, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    folder = os.path.expanduser(folder)
    if not (extensions is None) ^ (is_valid_file is None):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def check_file(x):
            return has_file_allowed_extension(x, extensions)
    else:
        check_file = is_valid_file
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(folder, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if check_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class CoMatchDatasetImageNetTest:
    """CoMatch Dataset for train."""
    def __init__(self, args, imgdir):
        self.args = args
        self.root = imgdir
        _, class_to_idx = CoMatchDatasetImageNetTest._find_classes(self.root)
        samples = make_dataset_test(self.root, class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=None)

        if not samples:
            raise RuntimeError("Found 0 files in subfolders of: " + self.root + "\n")

        self.samples = samples
        logging.info("sample len: %d", len(self.samples))

        # for test
        self.resize = vision.Resize(256)
        self.center_crop = vision.CenterCrop(224)
        self.to_tensor = vision.ToTensor()
        self.normalize = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)

    def __getitem__(self, index):
        """
       Args:
           index (int): Index
       Returns:
           tuple: (sample, target, path)
           sample : [label,[unlabel1_weak, unlabel2_strong0, unlabel2_strong1] ... ]
           target : [label_target, unlabel1_target, ...]
           path : [label_path, unlabel1_path, ...] //for debug//
       """
        label_path, label_target = self.samples[index]
        label_sample = Image.open(label_path).convert('RGB')

        # test data transform
        label_sample_resize = self.resize(label_sample)
        label_sample_center_crop = self.center_crop(label_sample_resize)
        label_sample_tensor = self.to_tensor(label_sample_center_crop)
        label_sample_normalize = self.normalize(label_sample_tensor)

        return label_sample_normalize, label_target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _find_classes(folder):
        """
        Finds the class folders in a dataset.

        Args:
            folder (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (folder), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(folder) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class CoMatchSelectSample:
    """CoMatch Dataset for train."""
    def __init__(self, annotation, aug_num):
        self.aug_num = aug_num
        samples = json.load(open(annotation, 'r'))["unlabeled_samples"]

        if not samples:
            raise RuntimeError("Found 0 files in subfolders of: " + self.root + "\n")

        self.samples = samples

        # for test
        self.random_resize_crop = vision.RandomResizedCrop(224, scale=(0.2, 1.))
        self.random_horizontal_flip = vision.RandomHorizontalFlip()

        self.to_tensor = vision.ToTensor()
        self.normalize = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)

    def __getitem__(self, index):
        """
       Args:
           index (int): Index
       Returns:
           tuple: (sample, target, path)
           sample : [label,[unlabel1_weak, unlabel2_strong0, unlabel2_strong1] ... ]
           target : [label_target, unlabel1_target, ...]
           path : [label_path, unlabel1_path, ...] //for debug//
       """
        label_path, label_target = self.samples[index]
        label_sample = Image.open(label_path).convert('RGB')

        # test data transform
        img_data = []
        for _ in range(self.aug_num):
            label_sample_resize = self.random_resize_crop(label_sample)
            label_sample_horizontal_flip = self.random_horizontal_flip(label_sample_resize)
            label_sample_tensor = self.to_tensor(label_sample_horizontal_flip)
            label_sample_normalize = self.normalize(label_sample_tensor)
            img_data.append(label_sample_normalize)

        return img_data, label_target, label_path

    def __len__(self):
        return len(self.samples)


def create_select_dataset(args):
    """Create dataset for CoMatch."""
    comatch_dataset = CoMatchSelectSample(args.annotation, args.aug_num)
    args.dataset_size = len(comatch_dataset)
    ds = de.GeneratorDataset(comatch_dataset, column_names=["img_data", "label_target", "label_path"],
                             shuffle=True, num_parallel_workers=1, shard_id=args.rank, num_shards=args.device_num)

    ds = ds.batch(args.batch_size, num_parallel_workers=1, drop_remainder=True)
    ds = ds.map(operations=[concate_data], input_columns=["img_data"])

    return ds, len(comatch_dataset)


def concate_data(img_data):
    batch_size = img_data.shape[0]
    if batch_size == 1:
        img = img_data[0]
    else:
        img_data_ = img_data[0]
        for bts in range(1, batch_size):
            img_data_ = np.concatenate((img_data_, img_data[bts]), axis=0)
        img = img_data_
    return img
