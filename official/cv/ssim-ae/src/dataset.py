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
import math
import time
import random
from glob import glob
import numpy as np
from PIL import Image


import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_trans
from mindspore.dataset.transforms.py_transforms import Compose

from model_utils.options import Options


def read_img(img_path):
    image = Image.open(img_path)
    return np.asarray(image)


class DataFlow():
    def __init__(self, filenames, grayscale):
        self.filenames = filenames
        self.grayscale = grayscale

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        batch_x = read_img(batch_x)
        batch_x = batch_x / 255.
        if self.grayscale:
            batch_x = np.expand_dims(batch_x, axis=-1)
        return (batch_x,)


class Dataloader():
    def __init__(self):
        self.config = Options().parse()

    def create_dataset(self, rank_size, rank_id):
        if self.config["do_aug"]:
            dataset_dir = self.config["aug_dir"]
            notice_copy_over = os.path.join(self.config["checkpoint_dir"], "copy_is_over")
            if rank_id == 0:
                print("Augmenting data...")
                self.__augment_images__()
                os.mkdir(notice_copy_over)
            while True:
                if os.path.exists(notice_copy_over):
                    break
                time.sleep(1)
        else:
            dataset_dir = self.config["train_data_dir"]
        file_list = glob(dataset_dir + '/*')
        ds_train_images = DataFlow(file_list, self.config["grayscale"])
        ds_train_images = ds.GeneratorDataset(ds_train_images, ["input"],
                                              shuffle=True, num_shards=rank_size,
                                              shard_id=rank_id)
        train_image_ds = ds_train_images.map(operations=self.__get_compose__(),
                                             input_columns=["input"])
        train_image_ds = train_image_ds.map(operations=postprocess
                                            , input_columns=["input"])
        train_image_ds = train_image_ds.batch(self.config["batch_size"], drop_remainder=True)

        return train_image_ds

    def __generate_image_list__(self):
        filenames = os.listdir(self.config["train_data_dir"])
        cfg = self.config["data_augment"]
        num_img = len(filenames)
        num_ave_aug = int(math.floor(cfg["augment_num"] / num_img))
        rem = cfg["augment_num"] - num_ave_aug * num_img
        lucky_seq = [True] * rem + [False] * (num_img - rem)
        random.shuffle(lucky_seq)
        img_list = [
            (os.sep.join([self.config["train_data_dir"], filename]),
             num_ave_aug + 1 if lucky else num_ave_aug)
            for filename, lucky in zip(filenames, lucky_seq)
        ]
        return img_list

    def __get_compose__(self):
        cfg = self.config["data_augment"]
        if self.config["do_aug"]:
            transforms_list = [
                c_trans.RandomCrop(cfg["crop_size"]),
                c_trans.RandomRotation(cfg["rotate_angle"]),
                c_trans.RandomHorizontalFlip(cfg["p_horizontal_flip"]),
                c_trans.RandomVerticalFlip(cfg["p_vertical_flip"])
            ]
            compose_trans = Compose(transforms_list)
        else:
            transforms_list = [
                c_trans.Resize(size=(cfg["im_resize"], cfg["im_resize"]))
            ]
            compose_trans = Compose(transforms_list)
        return compose_trans

    def __augment_images__(self):
        file_list = self.__generate_image_list__()
        cfg = self.config["data_augment"]
        for filepath, n in file_list:
            image = Image.open(filepath)
            if np.asarray(image).shape[:2] != (cfg["im_resize"], cfg["im_resize"]):
                image = image.resize((cfg["im_resize"], cfg["im_resize"]))
            file_name = filepath.split(os.sep)[-1]
            dot_pos = file_name.rfind('.')
            image_name = file_name[:dot_pos]
            ext = file_name[dot_pos:]
            for i in range(n):
                image_varied = image.copy()
                varied_image_name = '{}_{:0>4d}'.format(image_name, i)
                output_filepath = os.sep.join([
                    self.config["aug_dir"],
                    '{}{}'.format(varied_image_name, ext)])
                image_varied.save(output_filepath)


def postprocess(input_batch):
    res = np.transpose(input_batch, (2, 1, 0)).astype(np.float32)
    return res


def get_patch(image, new_size, stride):
    height, weighe = image.shape[:2]
    i, j = new_size, new_size
    patch = []
    while i <= height:
        while j <= weighe:
            patch.append(image[i - new_size:i, j - new_size:j])
            j += stride
        j = new_size
        i += stride
    return np.array(patch)


def patch2img(patches, im_size, patch_size, stride):
    img = np.zeros((im_size, im_size, patches.shape[3] + 1))
    i, j = patch_size, patch_size
    k = 0
    while i <= im_size:
        while j <= im_size:
            img[i - patch_size:i, j - patch_size:j, :-1] += patches[k]
            img[i - patch_size:i, j - patch_size:j, -1] += np.ones((patch_size, patch_size))
            k += 1
            j += stride
        j = patch_size
        i += stride
    mask = np.repeat(img[:, :, -1][..., np.newaxis], patches.shape[3], 2)
    img = img[:, :, :-1] / mask
    return img
