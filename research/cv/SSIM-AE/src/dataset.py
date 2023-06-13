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
import numpy as np
import cv2

import mindspore.dataset as ds
import mindspore.dataset.vision as c_trans

from model_utils.config import config as cfg
from src.utils import read_img, get_file_list


def random_crop(image, new_size):
    h, w = image.shape[:2]
    y = np.random.randint(0, h - new_size)
    x = np.random.randint(0, w - new_size)
    image = image[y : y + new_size, x : x + new_size]
    return image


def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    if crop:
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0
        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator
        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        img_rotated = img_rotated[y0 : y0 + h_crop, x0 : x0 + w_crop]
    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False
    if np.random.random() > p_crop:
        crop = False
    return rotate_image(img, angle, crop)


class DataFlow:
    def __init__(self, filenames, grayscale):
        self.filenames = filenames
        self.grayscale = grayscale

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        batch_x = read_img(batch_x, self.grayscale).astype(np.float32)
        batch_x = batch_x / 255.0
        if self.grayscale:
            batch_x = np.expand_dims(batch_x, axis=-1)
        return (batch_x,)


class Dataloader:
    def __init__(self):
        self.config = cfg
        filenames = get_file_list(self.config.train_data_dir, self.config.img_suffix)
        self.num_img = len(filenames)

    def create_dataset(self, rank_size, rank_id):
        if self.config.do_aug:
            if self.config.online_aug:
                dataset_dir = self.config.train_data_dir
                repeat = int(math.ceil(self.config.augment_num / self.num_img))
            else:
                repeat = 1
                dataset_dir = self.config.aug_dir
                notice_copy_over = os.path.join(self.config.tmp, "copy_is_over")
                if rank_id == 0:
                    print("Augmenting data...")
                    self._augment_images()
                    if not os.path.exists(notice_copy_over):
                        os.mkdir(notice_copy_over)
                while True:
                    if os.path.exists(notice_copy_over):
                        break
                    time.sleep(1)
        else:
            dataset_dir = self.config.train_data_dir
            repeat = int(math.ceil(self.config.augment_num / self.num_img))
        file_list = get_file_list(dataset_dir, self.config.img_suffix)
        ds_train_images = DataFlow(file_list, self.config.grayscale)
        ds_train_images = ds.GeneratorDataset(
            ds_train_images, ["input"], shuffle=True, num_shards=rank_size, shard_id=rank_id
        )
        train_image_ds = ds_train_images.map(operations=self._get_compose(), input_columns=["input"])
        if repeat != 1:
            train_image_ds = train_image_ds.repeat(repeat)
            train_image_ds = train_image_ds.shuffle(buffer_size=16 * self.config.batch_size)
        train_image_ds = train_image_ds.batch(self.config.batch_size, drop_remainder=True)
        return train_image_ds

    def _generate_image_list(self):
        filenames = get_file_list(self.config.train_data_dir, self.config.img_suffix)
        num_ave_aug = int(math.floor(self.config.augment_num / self.num_img))
        rem = self.config.augment_num - num_ave_aug * self.num_img
        lucky_seq = [True] * rem + [False] * (self.num_img - rem)
        np.random.shuffle(lucky_seq)
        img_list = [
            (filename, num_ave_aug + 1 if lucky else num_ave_aug) for filename, lucky in zip(filenames, lucky_seq)
        ]
        return img_list

    def _get_compose(self):
        if self.config.do_aug and self.config.online_aug:
            transforms_list = [
                c_trans.Resize(size=(self.config.im_resize, self.config.im_resize)),
                c_trans.RandomCrop(self.config.crop_size),
                c_trans.RandomRotation(self.config.rotate_angle),
                c_trans.RandomHorizontalFlip(self.config.p_horizontal_flip),
                c_trans.RandomVerticalFlip(self.config.p_vertical_flip),
                c_trans.HWC2CHW(),
            ]
        elif self.config.do_aug:
            transforms_list = [c_trans.HWC2CHW()]
        else:
            transforms_list = [c_trans.Resize(size=(self.config.im_resize, self.config.im_resize)), c_trans.HWC2CHW()]
        return transforms_list

    def _augment_images(self):
        file_list = self._generate_image_list()
        for filepath, n in file_list:
            img = read_img(filepath, self.config.grayscale)
            if img.shape[:2] != (self.config.im_resize, self.config.im_resize):
                img = cv2.resize(img, (self.config.im_resize, self.config.im_resize))
            filename = filepath.split(os.sep)[-1]
            dot_pos = filename.rfind(".")
            imgname = filename[:dot_pos]
            ext = filename[dot_pos:]

            print("Augmenting {} {} times".format(filename, n))
            for i in range(n):
                img_varied = img.copy()
                varied_imgname = "{}_{:0>3d}_".format(imgname, i)

                if np.random.random() < self.config.p_ratate:
                    img_varied_ = random_rotate(img_varied, self.config.rotate_angle, self.config.p_crop)
                    if img_varied_.shape[0] >= self.config.crop_size and img_varied_.shape[1] >= self.config.crop_size:
                        img_varied = img_varied_
                    varied_imgname += "r"

                if np.random.random() < self.config.p_crop:
                    img_varied = random_crop(img_varied, self.config.crop_size)
                    varied_imgname += "c"

                if np.random.random() < self.config.p_horizontal_flip:
                    img_varied = cv2.flip(img_varied, 1)
                    varied_imgname += "h"

                if np.random.random() < self.config.p_vertical_flip:
                    img_varied = cv2.flip(img_varied, 0)
                    varied_imgname += "v"

                output_filepath = os.sep.join([self.config.aug_dir, "{}{}".format(varied_imgname, ext)])
                cv2.imwrite(output_filepath, img_varied)


def postprocess(input_batch):
    res = np.transpose(input_batch, (2, 1, 0)).astype(np.float32)
    return res
