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
"""
MiniImageNet
"""
import os
import pickle
import numpy as np
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.transforms import Compose
from PIL import Image


class MiniImageNet:
    """
    MiniImageNet
    """

    def __init__(self, root_path, split='train'):
        self.split = split

        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']

        min_label = min(label)
        print("min_label", min_label)
        label = [x - min_label for x in label]

        image_size = 84
        normalize = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        if split == 'train':
            self.transforms = Compose([
                decode,
                vision.RandomCrop(image_size, padding=4),
                vision.ToTensor(),
                normalize
            ])
        else:
            self.transforms = Compose([
                decode,
                vision.Resize(image_size),
                vision.ToTensor(),
                normalize
            ])
        data = [self.transforms(x)[0] for x in data]
        self.len = len(data)
        self.data = np.array(data)
        self.label = np.array(label)
        self.n_classes = max(self.label) + 1

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.data[i], self.label[i]


def decode(img):
    """
    :param img:
    :return:
    """
    return Image.fromarray(img)
