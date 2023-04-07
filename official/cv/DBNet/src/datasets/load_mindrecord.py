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
# This file refers to the project https://github.com/MhLiao/DB.git

"""DBNet Dataset DataLoader"""
import os
import math
import glob
import time
import cv2
import numpy as np

from mindspore import dataset as ds
from mindspore.mindrecord import FileWriter

from .pre_process import MakeSegDetectionData, MakeBorderMap
from .random_thansform import RandomAugment


def get_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float32')
    return img


def scale_pad(img, polys, eval_size):
    """scale image and polys with short side, then pad to eval_size."""
    h, w, c = img.shape
    s_h = eval_size[0] / h
    s_w = eval_size[1] / w
    scale = min(s_h, s_w)
    new_h = int(scale * h)
    new_w = int(scale * w)
    img = cv2.resize(img, (new_w, new_h))
    padimg = np.zeros((eval_size[0], eval_size[1], c), img.dtype)
    padimg[:new_h, :new_w, :] = img
    polys = polys * scale
    return padimg, polys


def get_bboxes(gt_path, config):
    """Get polys and it's `dontcare` flag by gt_path."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    polys = []
    dontcare = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            dontcare.append(True)
        else:
            dontcare.append(False)
        if config.dataset.is_icdar2015:
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), np.array(dontcare)


def resize(img, polys=None, denominator=32, is_train=True):
    """Resize image and its polys."""
    w_scale = math.ceil(img.shape[1] / denominator) * denominator / img.shape[1]
    h_scale = math.ceil(img.shape[0] / denominator) * denominator / img.shape[0]
    img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
    if polys is None:
        return img
    if is_train:
        new_polys = []
        for poly in polys:
            poly[:, 0] = poly[:, 0] * w_scale
            poly[:, 1] = poly[:, 1] * h_scale
            new_polys.append(poly)
        polys = new_polys
    else:
        polys[:, :, 0] = polys[:, :, 0] * w_scale
        polys[:, :, 1] = polys[:, :, 1] * h_scale
    return img, polys


class IC15DataLoader():
    """IC15 DataLoader"""

    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train

        self.ra = RandomAugment(max_tries=config.dataset.random_crop.max_tries,
                                min_crop_side_ratio=config.dataset.random_crop.min_crop_side_ratio,
                                crop_size=self.config.train.train_size)
        self.ms = MakeSegDetectionData(config.train.min_text_size,
                                       config.train.shrink_ratio)
        self.mb = MakeBorderMap(config.train.shrink_ratio,
                                config.train.thresh_min, config.train.thresh_max)
        self.img_dir = config.train.img_dir if is_train else config.eval.img_dir
        self.gt_dir = config.train.gt_dir if is_train else config.eval.gt_dir

        self.img_paths = glob.glob(os.path.join(self.img_dir, '*' + str(config.train.img_format)))
        self.gt_paths = [os.path.join(self.gt_dir, 'gt_' + img_path.split('/')[-1].split('.')[0] + '.txt') for img_path
                         in self.img_paths]
        assert self.img_paths and self.gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        polys, dontcare = get_bboxes(gt_path, self.config)
        return img_path, polys, dontcare.astype(np.int32)

    def processing_train(self, img, polys, dontcare):
        # Random Augment
        dontcare = dontcare.astype(np.bool_)
        if self.config.train.is_transform:
            img, polys = self.ra.random_scale(img, polys, self.config.dataset.short_side)
            img, polys = self.ra.random_rotate(img, polys, self.config.dataset.random_angle)
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        img, polys = resize(img, polys, is_train=self.is_train)

        img, gt, gt_mask = self.ms.process(img, polys, dontcare)
        img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        return img, gt, gt_mask, thresh_map, thresh_mask

    def processing_test(self, img, polys, dontcare):
        original = resize(img)
        dontcare = dontcare.astype(np.bool_)
        polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))
        img, polys = resize(img, polys, is_train=self.is_train)

        polys = np.array(polys)
        dontcare = np.array(dontcare, dtype=np.bool8)
        img, polys = scale_pad(img, polys, self.config.eval.eval_size)
        return original, img, polys, dontcare


def data_to_mindrecord(mindrecord_path, dataset, file_num=8):
    """Create MindRecord file."""
    writer = FileWriter(mindrecord_path, file_num)
    data_json = {
        "img": {"type": "bytes"},
        "polys": {"type": "int64", "shape": [-1, 8]},
        "dontcare": {"type": "int32", "shape": [-1]},
    }
    writer.add_schema(data_json, "data_json")
    data_iter = dataset.create_dict_iterator(output_numpy=True)

    for data in data_iter:
        with open(str(data["img_path"]), 'rb') as f:
            img = f.read()
        row = {"img": img, "polys": data["polys"], "dontcare": data["dontcare"]}
        writer.write_raw_data([row])
    writer.commit()


def save_mindrecord(config):
    dataset_dir = os.path.join(config.mindrecord_path, "ic15")
    os.makedirs(dataset_dir, exist_ok=True)
    if not hasattr(config, "device_num"):
        config.device_num = 1
    if not hasattr(config, "rank_id"):
        config.rank_id = 0
    train_data_path = os.path.join(dataset_dir, "ic15_train.md")
    eval_data_path = os.path.join(dataset_dir, "ic15_eval.md")
    if config.rank_id == 0:
        if not os.path.exists(train_data_path + "0.db"):
            config.logger.info(f"create train dataset mindrecord, at {dataset_dir}")
            train_data_loader = IC15DataLoader(config, is_train=True)
            train_dataset = ds.GeneratorDataset(train_data_loader, ['img_path', 'polys', 'dontcare'])
            data_to_mindrecord(train_data_path, train_dataset, file_num=8)
        if not os.path.exists(eval_data_path + ".db"):
            config.logger.info(f"create eval dataset mindrecord, at {dataset_dir}")
            eval_data_loader = IC15DataLoader(config, is_train=False)
            eval_dataset = ds.GeneratorDataset(eval_data_loader, ['img_path', 'polys', 'dontcare'])
            data_to_mindrecord(eval_data_path, eval_dataset, file_num=1)
    else:
        while True:
            if os.path.exists(train_data_path + "0.db") and  os.path.exists(eval_data_path + ".db"):
                break
            time.sleep(1)
    config.train_mindrecord = train_data_path
    config.eval_mindrecord = eval_data_path


def create_dataset_md(config, is_train):
    """Create MindSpore Dataset object from mindrecord."""
    save_mindrecord(config)
    ds.config.set_prefetch_size(config.dataset.prefetch_size)
    ds.config.set_enable_shared_mem(True)
    data_loader = IC15DataLoader(config, is_train=is_train)
    change_swap_op = ds.vision.HWC2CHW()
    normalize_op = ds.vision.Normalize(mean=config.dataset.mean, std=config.dataset.std)
    color_adjust_op = ds.vision.RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
    if is_train:
        dataset = ds.MindDataset([config.train_mindrecord + str(i) for i in range(8)],
                                 columns_list=['img', 'polys', 'dontcare'],
                                 num_shards=config.device_num, shard_id=config.rank_id, shuffle=True,
                                 num_parallel_workers=8)
        dataset = dataset.map(operations=ds.vision.Decode(), input_columns=["img"])
        dataset = dataset.map(operations=data_loader.processing_train, input_columns=['img', 'polys', 'dontcare'],
                              output_columns=['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
                              python_multiprocessing=True, num_parallel_workers=config.dataset.num_workers)
        dataset = dataset.map(operations=[color_adjust_op, normalize_op, change_swap_op], input_columns=["img"])
        dataset = dataset.project(['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'])
    else:
        dataset = ds.MindDataset([config.eval_mindrecord], columns_list=['img', 'polys', 'dontcare'],
                                 num_parallel_workers=8)
        dataset = dataset.map(operations=ds.vision.Decode(), input_columns=["img"])
        dataset = dataset.map(operations=data_loader.processing_test, input_columns=['img', 'polys', 'dontcare'],
                              output_columns=['original', 'img', 'polys', 'dontcare'],
                              python_multiprocessing=True, num_parallel_workers=config.dataset.num_workers)
        dataset = dataset.map(operations=[normalize_op, change_swap_op], input_columns=["img"])
        dataset = dataset.project(['original', 'img', 'polys', 'dontcare'])
    batch_size = config.train.batch_size if is_train else 1
    dataset = dataset.batch(batch_size, drop_remainder=is_train, num_parallel_workers=config.dataset.num_workers)
    steps_pre_epoch = dataset.get_dataset_size()
    return dataset, steps_pre_epoch
