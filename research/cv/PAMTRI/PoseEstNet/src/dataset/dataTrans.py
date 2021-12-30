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
"""dataTrans"""
import os
import csv
import cv2
import numpy as np

from .transforms import get_affine_transform

class VeRiTransDataset():
    """VeRiTransDataset"""
    def __init__(self, cfg, root, image_set):
        super().__init__()
        self.root = root
        self.image_set = image_set
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.flip_pairs = [[0, 18], [1, 19], [2, 20], [3, 21], [4, 22], [5, 23], [6, 24], [7, 25], [8, 26], [9, 27],
                           [10, 28], [11, 29], [12, 30], [13, 31], [14, 32], [15, 33], [16, 34], [17, 35]]
        self.image_width = 256
        self.image_height = 256
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.db = self._get_db()

    def _get_db(self):
        """_get_db"""
        file_name = os.path.join(
            self.root, 'label_' + self.image_set + '.csv'
        )

        image_name = []
        with open(file_name) as annot_file:
            reader = csv.reader(annot_file, delimiter=',')
            for row in reader:
                image_name.append(row[0])

        gt_db = []
        for k in range(len(image_name)):
            name = os.path.join(self.root, 'image_' + self.image_set, image_name[k])

            data_numpy = cv2.imread(
                name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

            if data_numpy is None:
                raise ValueError('Fail to read {}'.format(name))

            height, width, _ = data_numpy.shape

            if self.color_rgb:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

            center = np.zeros((2), dtype=np.float32)
            center[0] = width * 0.5
            center[1] = height* 0.5

            if width > self.aspect_ratio * height:
                height = width * 1.0 / self.aspect_ratio
            elif width < self.aspect_ratio * height:
                width = height * self.aspect_ratio

            scale = np.array(
                [width * 1.0 / self.pixel_std, height * 1.0 / self.pixel_std],
                dtype=np.float32)
            if center[0] != -1:
                scale = scale * 1.25

            c = center
            s = scale
            r = 0

            trans = get_affine_transform(c, s, r, self.image_size)

            _input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

            gt_db.append(
                {
                    'image': _input,
                    'center': center,
                    'scale': scale,
                }
            )

        return gt_db

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = self.db[idx]

        return db_rec['image'], db_rec['center'], db_rec['scale']
