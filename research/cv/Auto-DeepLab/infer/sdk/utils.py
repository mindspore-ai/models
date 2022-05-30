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
# ===========================================================================
"""sdk utils"""
import os
import numpy as np
from PIL import Image


cityspallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]


def fast_hist(predict, label, n):
    """
    fast_hist
    inputs:
        - predict (ndarray)
        - label (ndarray)
        - n (int) - number of classes
    outputs:
        - fast histogram
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(np.int32) + predict[k], minlength=n ** 2).reshape(n, n)


def encode_segmap(lbl, ignore_label):
    """encode segmap"""
    mask = np.uint8(lbl)

    num_classes = 19
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    class_map = dict(zip(valid_classes, range(num_classes)))
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_label
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]

    return mask


def label_to_color_image(npimg) -> Image:
    """label_to_color_image"""
    img = Image.fromarray(npimg.astype('uint8'), "P")
    img.putpalette(cityspallete)
    out_img = img.convert('RGB')
    return out_img


class CityscapesDataLoader():
    """CityscapesDataLoader"""
    def __init__(self, data_root, split='val', ignore_label=255):
        """__init__"""
        super(CityscapesDataLoader, self).__init__()
        images_base = os.path.join(data_root, 'leftImg8bit', split)
        annotations_base = os.path.join(data_root, 'gtFine', split)
        self.img_id = []
        self.img_path = []
        self.img_name = []
        self.images = []
        self.gtFiles = []
        for root, _, files in os.walk(images_base):
            for filename in files:
                if filename.endswith('.png'):
                    self.img_path.append(root)
                    self.img_name.append(filename)
                    folder_name = root.split(os.sep)[-1]
                    gtFine_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    _img = os.path.join(root, filename)
                    _gt = os.path.join(annotations_base, folder_name, gtFine_name)
                    self.images.append(_img)
                    self.gtFiles.append(_gt)
        self.len = len(self.images)
        self.cur_index = 0
        print(f"Found {self.len} images")

    def __len__(self):
        """__len__"""
        return self.len

    def __iter__(self):
        """__iter__"""
        return self

    def __next__(self) -> dict:
        """__next__"""
        if self.cur_index == self.len:
            raise StopIteration()

        with open(self.images[self.cur_index], 'rb') as f:
            image = f.read()
        gtFine = Image.open(self.gtFiles[self.cur_index])
        gtFine = np.array(gtFine).astype(np.uint8)
        dataItem = {
            'file_path': self.img_path[self.cur_index],
            'file_name': self.img_name[self.cur_index],
            'img': image,
            'gt': gtFine,
        }
        self.cur_index += 1
        return dataItem
