# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Dataset LIP generator."""
import os
import cv2
import numpy as np

from src.dataset.basedataset import BaseDataset


class LIP(BaseDataset):
    """Dataset LIP generator."""
    def __init__(self,
                 root,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=None,
                 downsample_rate=1,
                 scale_factor=11,
                 mean=None,
                 std=None,
                 is_train=True):

        super(LIP, self).__init__(ignore_label=ignore_label, base_size=base_size,
                                  crop_size=crop_size, downsample_rate=downsample_rate,
                                  scale_factor=scale_factor, mean=mean, std=std)
        self.root = root
        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        if is_train:
            self.list_path = root + "/train.lst"
        else:
            self.list_path = root + "/val.lst"
        self.img_list = [line.strip().split() for line in open(self.list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        """Read each line in lst file."""
        files = []
        for item in self.img_list:
            if 'train' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name}
            elif 'val' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name}
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        """Resize image."""
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(self.root, 'TrainVal_images/', item["img"]),
                           cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, 'TrainVal_parsing_annotations/', item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        size = label.shape

        if 'testval' in self.list_path:
            image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]
        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label,
                                       self.multi_scale, False)

        return image.copy(), label.copy()
