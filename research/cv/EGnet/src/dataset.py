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

"""Create train or eval dataset."""

import os
import random
from PIL import Image
import cv2
import numpy as np

from model_utils.config import base_config
from mindspore.dataset import GeneratorDataset
from mindspore.communication.management import get_rank, get_group_size

if base_config.train_online:
    import moxing as mox

    mox.file.shift('os', 'mox')


class ImageDataTrain:
    """
    training dataset
    """

    def __init__(self, train_path=""):
        self.sal_root = train_path
        self.sal_source = os.path.join(train_path, "train_pair_edge.lst")
        with open(self.sal_source, "r") as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        sal_image = load_image(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[0]))
        sal_label = load_sal_label(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[1]))
        sal_edge = load_edge_label(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[2]))
        sal_image, sal_label, sal_edge = cv_random_flip(sal_image, sal_label, sal_edge)
        return sal_image, sal_label, sal_edge

    def __len__(self):
        return self.sal_num


class ImageDataTest:
    """
    test dataset
    """

    def __init__(self, test_mode=1, sal_mode="e", test_path="", test_fold=""):
        if test_mode == 1:
            if sal_mode == "e":
                self.image_root = test_path + "/ECSSD/images/"
                self.image_source = test_path + "/ECSSD/test.lst"
                self.test_fold = test_fold + "/ECSSD/"
                self.test_root = test_path + "/ECSSD/ground_truth_mask/"
            elif sal_mode == "p":
                self.image_root = test_path + "/PASCAL-S/images/"
                self.image_source = test_path + "/PASCAL-S/test.lst"
                self.test_fold = test_fold + "/PASCAL-S/"
                self.test_root = test_path + "/PASCAL-S/ground_truth_mask/"
            elif sal_mode == "d":
                self.image_root = test_path + "/DUT-OMRON/images/"
                self.image_source = test_path + "/DUT-OMRON/test.lst"
                self.test_fold = test_fold + "/DUT-OMRON/"
                self.test_root = test_path + "/DUT-OMRON/ground_truth_mask/"
            elif sal_mode == "h":
                self.image_root = test_path + "/HKU-IS/images/"
                self.image_source = test_path + "/HKU-IS/test.lst"
                self.test_fold = test_fold + "/HKU-IS/"
                self.test_root = test_path + "/HKU-IS/ground_truth_mask/"
            elif sal_mode == "s":
                self.image_root = test_path + "/SOD/images/"
                self.image_source = test_path + "/SOD/test.lst"
                self.test_fold = test_fold + "/SOD/"
                self.test_root = test_path + "/SOD/ground_truth_mask/"
            elif sal_mode == "t":
                self.image_root = test_path + "/DUTS-TE/DUTS-TE-Image"
                self.image_source = test_path + "/DUTS-TE/test.lst"
                self.test_fold = test_fold + "/DUTS-TE/"
                self.test_root = test_path + "/DUTS-TE/DUTS-TE-Mask/"
            else:
                raise ValueError("Unknown sal_mode")
        else:
            raise ValueError("Unknown sal_mode")

        with open(self.image_source, "r") as f:
            self.image_list = [x.strip() for x in f.readlines()]
        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, _ = load_image_test(os.path.join(self.image_root, self.image_list[item]))
        label = load_sal_label(os.path.join(self.test_root, self.image_list[item][0:-4] + ".png"))
        return image, label, item % self.image_num

    def save_folder(self):
        return self.test_fold

    def __len__(self):
        return self.image_num


# get the dataloader (Note: without data augmentation, except saliency with random flip)
def create_dataset(batch_size, mode="train", num_thread=1, test_mode=1, sal_mode="e", train_path="", test_path="",
                   test_fold="", is_distributed=False, rank_id=0, rank_size=1):
    """
    create dataset
    """
    shuffle = False
    drop_remainder = False

    if mode == "train":
        shuffle = True
        drop_remainder = True
        dataset = ImageDataTrain(train_path=train_path)
    else:
        dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode, test_path=test_path, test_fold=test_fold)

    if is_distributed:
        # get rank_id and rank_size
        rank_id = get_rank()
        rank_size = get_group_size()
        ds = GeneratorDataset(dataset, column_names=["sal_image", "sal_label", "sal_edge_or_index"],
                              shuffle=shuffle, num_parallel_workers=num_thread, num_shards=rank_size, shard_id=rank_id)
    else:
        ds = GeneratorDataset(dataset, column_names=["sal_image", "sal_label", "sal_edge_or_index"],
                              shuffle=shuffle, num_parallel_workers=num_thread)
    return ds.batch(batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_thread), dataset


def save_img(img, path, is_distributed=False):

    if is_distributed and get_rank() != 0:
        return
    range_ = np.max(img) - np.min(img)
    img = (img - np.min(img)) / range_
    img = img * 255 + 0.5
    img = img.astype(np.uint8).squeeze()
    Image.fromarray(img).save(path)


def load_image(pah):
    if not os.path.exists(pah):
        print("File Not Exists,", pah)
    im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_image_test(pah):
    """
    load test image
    """
    pah = pah.split(".")[0]
    if "HKU-IS" in pah:
        pah = pah + ".png"
    else:
        pah = pah + ".jpg"
    if not os.path.exists(pah):
        print("File Not Exists")
    im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_edge_label(pah):
    """
    pixels > 0.5 -> 1
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print("File Not Exists")
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = label / 255.
    label[np.where(label > 0.5)] = 1.
    label = label[np.newaxis, ...]
    return label


def load_sal_label(pah):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print("File Not Exists")
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
        edge = edge[:, :, ::-1].copy()
    return img, label, edge
