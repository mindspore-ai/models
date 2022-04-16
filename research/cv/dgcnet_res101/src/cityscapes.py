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
"""DGCNet(res101) dataset."""
import argparse
import os.path as osp
import random
import numpy as np
import cv2

from mindspore import dataset as ds


def str2bool(v):
    """str2bool"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        result = True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        result = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    return result


class Cityscapes:
    """Cityscapes Dataset for DGCNet(res101)."""
    def __init__(self, opts, crop_size, scale, mirror, max_iters, mean, vari, ignore_label=255):
        self.root = opts.data_dir
        self.list_path = opts.data_list
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.vari = vari
        self.is_mirror = mirror
        self.rgb = opts.rgb
        self.img_ids = [i_id.strip().split() for i_id in open(self.list_path)]
        self.max_iters = max_iters
        if self.max_iters is not None:
            self.img_ids = opts.multiple * opts.batch_size * self.img_ids * \
                           int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "image_path": image_path,
                "label_path": label_path
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        """len"""
        return len(self.files)

    def generate_scale_label(self, image, label):
        """generate_scale_label"""
        f_scale = 0.7 + random.randint(0, 14) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label, reverse=False):
        """id2trainId"""
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            (img, label) (tuple): Pairs of dataset
        """
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)

        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        if self.rgb:
            image = image[:, :, ::-1]  ## BGR -> RGB
            image /= 255         ## using pytorch pretrained models

        image -= self.mean
        image /= self.vari

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy()


def create_dataset(opts, crop_size, scale, mirror, max_iters, mean, vari, device_num=1, device_id=0):
    """create cityscapes dataset"""
    dataset = Cityscapes(opts, crop_size=crop_size, max_iters=max_iters, mean=mean, vari=vari, scale=scale,
                         mirror=mirror)
    DS = ds.GeneratorDataset(dataset, column_names=["image", "label"], num_parallel_workers=opts.num_workers,
                             shuffle=True, num_shards=device_num, shard_id=device_id)
    DS = DS.batch(opts.batch_size, num_parallel_workers=opts.num_workers, drop_remainder=True)
    return DS
