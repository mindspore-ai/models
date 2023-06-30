# Copyright 2023 Huawei Technologies Co., Ltd
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
import stat

import cv2
from mindspore.dataset import engine as de

from src.base_dataset import BaseDataset
from src.model_utils.common import rank_sync


class CitySpacesDataset(BaseDataset):

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 root_path,
                 base_size=(512, 1024),
                 crop_size=(512, 512),
                 ratio_range=(0.5, 2.0),
                 ignore_label=255,
                 ignore_image=0,
                 cat_max_ratio=0.75,
                 prob=0.5,
                 mean=None,
                 std=None,
                 is_train=True):

        super(CitySpacesDataset, self).__init__(base_size, crop_size,
                                                ratio_range, ignore_label,
                                                ignore_image, cat_max_ratio,
                                                prob, mean, std)

        self._index = 0
        self.root_path = root_path
        if is_train:
            self.list_path = os.path.join(root_path, "train.txt")
            # include 1. is_ratio_resize, 2. is_crop,
            # 3. is_flip, 4. is_photo_distortion, 5. is_pad
            self.special_operation = (True, True, True, True, True)
        else:
            self.list_path = os.path.join(root_path, "val.txt")
            self.special_operation = (False, False, False, False, False)
        with open(self.list_path) as f:
            img_list = [line.strip().split() for line in f]
        self.img_list = [(vector[0], vector[1]) for vector in img_list if len(vector) >= 2]
        self._number = len(self.img_list)

    def __len__(self):
        return self._number

    def __getitem__(self, index):
        if index < self._number:
            image_path = self.img_list[index][0]
            label_path = self.img_list[index][1]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            image, label = self.gen_sample(image, label, self.special_operation)
        else:
            raise StopIteration
        image_out = image.copy()
        label_out = label.copy()
        return image_out, label_out

    def show(self):
        for line in self.img_list:
            print(line)


@rank_sync
def prepare_cityscape_dataset(data_path):
    train_list_file = os.path.join(data_path, "train.txt")
    val_list_file = os.path.join(data_path, "val.txt")
    if os.path.exists(train_list_file) and os.path.exists(val_list_file):
        print(f"Dataset {train_list_file} and {val_list_file} already exists, use it.")
        return
    train_image_path = os.path.join(data_path, "leftImg8bit/train")

    val_image_path = os.path.join(data_path, "leftImg8bit/val")

    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(train_list_file, flags, modes), 'w') as train:
        for dir_path, _, filenames in os.walk(train_image_path):
            if filenames:
                for filename in filenames:
                    if filename.endswith("leftImg8bit.png"):
                        label_name = filename.replace("leftImg8bit.png", "gtFine_labelTrainIds.png")
                        label_path = os.path.join(dir_path.replace("leftImg8bit", "gtFine"), label_name)
                        if os.path.exists(label_path):
                            sample = os.path.join(dir_path, filename) + " " + os.path.join(label_path) + "\n"
                            train.write(sample)

    with os.fdopen(os.open(val_list_file, flags, modes), 'w') as val:
        for dir_path, _, filenames in os.walk(val_image_path):
            if filenames:
                for filename in filenames:
                    if filename.endswith("leftImg8bit.png"):
                        label_name = filename.replace("leftImg8bit.png", "gtFine_labelTrainIds.png")
                        label_path = os.path.join(dir_path.replace("leftImg8bit", "gtFine"), label_name)
                        if os.path.exists(label_path):
                            sample = os.path.join(dir_path, filename) + " " + os.path.join(label_path) + "\n"
                            val.write(sample)
    print(f"Dataset {train_list_file} and {val_list_file} create completed.")


def get_eval_dataset(config):
    eval_ori_dataset = CitySpacesDataset(root_path=config.data_path,
                                         ignore_label=config.dataset_ignore_label,
                                         ignore_image=config.dataset_ignore_image,
                                         base_size=config.base_size,
                                         crop_size=config.crop_size,
                                         mean=config.img_norm_mean,
                                         std=config.img_norm_std,
                                         is_train=False)
    eval_dataset = de.GeneratorDataset(eval_ori_dataset,
                                       column_names=["image", "label"],
                                       shuffle=False,
                                       num_parallel_workers=config.dataset_eval_num_parallel_workers,
                                       max_rowsize=config.dataset_max_rowsize,
                                       python_multiprocessing=False)
    eval_dataset = eval_dataset.batch(1, num_parallel_workers=config.dataset_eval_num_parallel_workers)
    return eval_dataset


def get_train_dataset(config):
    ori_dataset = CitySpacesDataset(root_path=config.data_path,
                                    ignore_label=config.dataset_ignore_label,
                                    ignore_image=config.dataset_ignore_image,
                                    base_size=config.base_size,
                                    crop_size=config.crop_size,
                                    mean=config.img_norm_mean,
                                    std=config.img_norm_std,
                                    is_train=True)
    if config.rank_size > 1:
        train_dataset = de.GeneratorDataset(ori_dataset,
                                            column_names=["image", "label"],
                                            shuffle=True,
                                            num_parallel_workers=config.dataset_num_parallel_workers,
                                            num_shards=config.rank_size,
                                            shard_id=config.rank_id,
                                            max_rowsize=config.dataset_max_rowsize,
                                            python_multiprocessing=False)
    else:
        train_dataset = de.GeneratorDataset(ori_dataset,
                                            column_names=["image", "label"],
                                            shuffle=True,
                                            num_parallel_workers=config.dataset_num_parallel_workers,
                                            max_rowsize=config.dataset_max_rowsize,
                                            python_multiprocessing=False)
    train_dataset = train_dataset.batch(config.batch_size, num_parallel_workers=config.dataset_num_parallel_workers,
                                        drop_remainder=True)
    return train_dataset
