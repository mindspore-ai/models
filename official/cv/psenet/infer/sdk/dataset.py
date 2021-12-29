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
import os
import cv2
import numpy as np
from PIL import Image


class testdataset:
    def __init__(self, test_root_dir='.', resize=1920):
        self.resize = resize
        self.test_root_dir = test_root_dir
        self.counter = 0

    def __iter__(self):
        self.imagedir = os.path.join(self.test_root_dir, 'ch4_test_images')

        if not os.path.exists(self.imagedir):
            raise ValueError("test dataset is not exist!")
        self.img_names = [i for i in os.listdir(self.imagedir) if
                          os.path.splitext(i)[-1].lower() in ['.jpg', '.png', '.jpeg']]

        self.image_paths = []
        for img_name in self.img_names:
            self.image_paths.append(os.path.join(self.imagedir, img_name))
        return self

    def __next__(self):
        if self.counter >= len(self.image_paths):
            raise StopIteration()
        img_path = self.image_paths[self.counter]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        long_size = max(img.shape[:2])
        img_resized = np.zeros((long_size, long_size, 3), np.uint8)
        img_resized[:img.shape[0], :img.shape[1], :] = img
        img_resized = cv2.resize(img_resized, dsize=(self.resize, self.resize))
        img_resized = Image.fromarray(img_resized)
        img_resized = img_resized.convert('RGB')
        img_resized = np.asarray(img_resized)
        img_name = os.path.split(self.image_paths[self.counter])[-1]
        self.counter += 1
        return img, img_resized, img_name
