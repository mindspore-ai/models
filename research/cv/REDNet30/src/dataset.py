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
"""dataset."""
import io
import random
import glob
import numpy as np
from PIL import Image


class Dataset():
    """Dataset."""
    def __init__(self, dataset_path, patch_size):
        self.image_files = sorted(glob.glob(dataset_path + '/*'))
        self.patch_size = patch_size

    def __getitem__(self, idx):
        label = Image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        crop_x = random.randint(0, label.width - self.patch_size)
        crop_y = random.randint(0, label.height - self.patch_size)
        label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        # additive jpeg noise
        buffer = io.BytesIO()
        label.save(buffer, format='jpeg', quality=10)
        input_img = Image.open(buffer)

        input_img = np.array(input_img).astype(np.float32)
        label = np.array(label).astype(np.float32)
        input_img = np.transpose(input_img, axes=[2, 0, 1])
        label = np.transpose(label, axes=[2, 0, 1])

        # normalization
        input_img /= 255.0
        label /= 255.0

        return input_img, label

    def __len__(self):
        return len(self.image_files)
