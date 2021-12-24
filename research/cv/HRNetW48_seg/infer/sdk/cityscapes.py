# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cityscapes dataset loader for SDK inference."""
import os
import cv2
import numpy as np


class Cityscapes:
    """Cityscapes dataset loader."""
    def __init__(self, root, lst):
        self.root = root
        self.lst = lst
        with open(self.lst, "r") as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            sample = line.strip().split()
            self.samples.append(sample)
        ignore_label = 255
        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}
        self.length = len(self.samples)

    def __getitem__(self, index):
        if index < self.length:
            img_path = os.path.join(self.root, self.samples[index][0])
            msk_path = os.path.join(self.root, self.samples[index][1])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            msk = self.convert_label(msk).astype(np.int32)
            img = np.expand_dims(img, axis=0)
            msk = np.expand_dims(msk, axis=0)
            print(f"= {index+1} = ", img_path)
            name = self.samples[index][0].split("/")[-1]
        else:
            raise StopIteration
        return img.copy(), msk.copy(), name

    def __len__(self):
        return self.length

    def convert_label(self, msk, inverse=False):
        """Convert classification ids in labels."""
        temp = msk.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                msk[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                msk[temp == k] = v
        return msk
