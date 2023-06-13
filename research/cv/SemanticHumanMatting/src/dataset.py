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

"""Load datasets"""
import os
import random as r

import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms as C
from mindspore import dtype as mstype


def safe_modify_file_name(file_name):
    if not os.path.exists(file_name):
        if "jpg" in file_name:
            return file_name.replace("jpg", "png")
        return file_name.replace("png", "jpg")

    return file_name


def read_files(data_dir, file_name):
    image_name = os.path.join(data_dir, "clip_img", file_name["image"].replace("matting", "clip").replace("png", "jpg"))
    trimap_name = os.path.join(data_dir, "trimap", file_name["trimap"].replace("clip", "matting"))
    alpha_name = os.path.join(data_dir, "alpha", file_name["alpha"].replace("clip", "matting"))

    image_name = safe_modify_file_name(image_name)
    trimap_name = safe_modify_file_name(trimap_name)
    alpha_name = safe_modify_file_name(alpha_name)

    image = cv2.imread(image_name)
    trimap = cv2.imread(trimap_name)
    alpha = cv2.imread(alpha_name)

    return image, trimap, alpha


def random_scale_and_creat_patch(image, trimap, alpha, patch_size):
    # random scale
    if r.random() < 0.5:
        h, w, _ = image.shape
        scale = 0.75 + 0.5 * r.random()
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    # create patch
    if r.random() < 0.5:
        h, w, _ = image.shape
        if h > patch_size and w > patch_size:
            x = r.randrange(0, w - patch_size)
            y = r.randrange(0, h - patch_size)
            image = image[y : y + patch_size, x : x + patch_size, :]
            trimap = trimap[y : y + patch_size, x : x + patch_size, :]
            alpha = alpha[y : y + patch_size, x : x + patch_size, :]
        else:
            image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            trimap = cv2.resize(trimap, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

    return image, trimap, alpha


def random_flip(image, trimap, alpha):
    if r.random() < 0.5:
        image = cv2.flip(image, 0)
        trimap = cv2.flip(trimap, 0)
        alpha = cv2.flip(alpha, 0)

    if r.random() < 0.5:
        image = cv2.flip(image, 1)
        trimap = cv2.flip(trimap, 1)
        alpha = cv2.flip(alpha, 1)
    return image, trimap, alpha


class HumanMattingData:
    """
    Load datasets
    """

    def __init__(self, root_dir, imglist, patch_size, train_mode):
        super().__init__()
        self.data_root = root_dir
        self.train_mode = train_mode
        self.patch_size = patch_size
        with open(imglist) as f:
            self.imgID = f.readlines()
        self.num = len(self.imgID)
        print("{} dataset : file number {}".format(train_mode, str(self.num)))

    def __getitem__(self, index):
        # read files
        image, trimap, alpha = read_files(
            os.path.join(self.data_root, self.train_mode),
            file_name={
                "image": self.imgID[index].strip(),
                "trimap": self.imgID[index].strip()[:-4] + ".png",
                "alpha": self.imgID[index].strip()[:-4] + ".png",
            },
        )
        # NOTE ! ! !
        # trimap should be 3 classes : fg, bg. unsure
        trimap[trimap == 0] = 0
        trimap[trimap >= 250] = 2
        trimap[np.where(~((trimap == 0) | (trimap == 2)))] = 1

        # augmentation
        if self.train_mode == "train":
            image, trimap, alpha = random_scale_and_creat_patch(image, trimap, alpha, self.patch_size)
            image, trimap, alpha = random_flip(image, trimap, alpha)
        else:
            image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            trimap = cv2.resize(trimap, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        # normalize
        image = (
            image.astype(np.float32)
            - (
                114.0,
                121.0,
                134.0,
            )
        ) / 255.0
        alpha = alpha.astype(np.float32) / 255.0

        trimap = np.expand_dims(trimap[:, :, 0], -1)
        alpha = np.expand_dims(alpha[:, :, 0], -1)

        image = np.transpose(image, (2, 0, 1))
        trimap = np.transpose(trimap, (2, 0, 1))
        alpha = np.transpose(alpha, (2, 0, 1))

        # 3 channel trimap
        trimap_dl = trimap.copy()
        trimap_dl[trimap == 0] = 1
        trimap_dl[trimap != 0] = 0
        trimap_ch0 = trimap_dl  # bg

        trimap_dl = trimap.copy()
        trimap_dl[trimap == 2] = 1
        trimap_dl[trimap != 2] = 0
        trimap_ch1 = trimap_dl  # fg

        trimap_dl = trimap.copy()
        trimap_dl[trimap == 1] = 1
        trimap_dl[trimap != 1] = 0
        trimap_ch2 = trimap_dl  # unsure

        trimap_ch = np.concatenate((trimap_ch0, trimap_ch1, trimap_ch2), axis=0)

        return image, trimap_ch, trimap, alpha

    def __len__(self):
        return self.num


def create_dataset(cfg, usage="train", repeat_num=1):
    imglist = os.path.join(cfg["dataDir"], usage, "{}.txt".format(usage))

    dataset_generator = HumanMattingData(cfg["dataDir"], imglist, cfg["patch_size"], usage)
    if cfg["group_size"] == 1:
        dataset = ds.GeneratorDataset(
            dataset_generator,
            ["image", "trimap_ch", "trimap", "alpha"],
            num_parallel_workers=cfg["nThreads"],
            shuffle=False,
            python_multiprocessing=True,
        )
    elif usage == "train":
        dataset = ds.GeneratorDataset(
            dataset_generator,
            ["image", "trimap_ch", "trimap", "alpha"],
            num_parallel_workers=cfg["nThreads"],
            shuffle=False,
            num_shards=cfg["group_size"],
            shard_id=cfg["rank"],
            python_multiprocessing=True,
        )
    else:
        sampler = ds.DistributedSampler(num_shards=cfg["group_size"], shard_id=cfg["rank"], shuffle=False, offset=0)
        dataset = ds.GeneratorDataset(
            dataset_generator,
            ["image", "trimap_ch", "trimap", "alpha"],
            num_parallel_workers=cfg["nThreads"],
            sampler=sampler,
            python_multiprocessing=True,
        )

    transform_image = C.TypeCast(mstype.float32)
    transform_trimap = C.TypeCast(mstype.int32)

    dataset = dataset.map(operations=transform_image, input_columns="image")
    dataset = dataset.map(operations=transform_image, input_columns="trimap_ch")
    dataset = dataset.map(operations=transform_trimap, input_columns="trimap")

    dataset = dataset.batch(cfg["train_batch"], drop_remainder=True)
    steps_per_epoch = dataset.get_dataset_size()
    dataset = dataset.repeat(repeat_num)

    return dataset, steps_per_epoch
