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
import csv
import numpy as np


class miniImageNet:
    def __init__(self, data_path, split_path, partition='train'):
        self.data_root = data_path
        self.partition = partition

        file_path = os.path.join(split_path, 'miniImageNet', '{}.csv'.format(self.partition))
        self.imgs, self.labels = self._read_csv(file_path)

    def _read_csv(self, file_path):
        imgs = []
        labels = []
        labels_name = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                img, label = row[0], row[1]
                img = os.path.join(self.data_root, 'images/{}'.format(img))
                imgs.append(img)
                if label not in labels_name:
                    labels_name.append(label)
                labels.append(labels_name.index(label))
        return imgs, labels

    def __getitem__(self, item):
        img = self.imgs[item]
        img = np.fromfile(img, np.uint8) #Image.open(img)
        target = self.labels[item]
        return img, target

    def __len__(self):
        return len(self.labels)
        