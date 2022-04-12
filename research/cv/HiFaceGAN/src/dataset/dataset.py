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
"""HiFaceGAN dataset"""
import random
from pathlib import Path

import cv2
import mindspore as ms
import mindspore.dataset as ds


class BaseDataset:
    """Base HiFaceGAN dataset"""

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        files = [str(x) for x in list(Path(self.data_path).glob('*.png'))]
        # path to concatenated images
        self.img_path = files
        self.dataset_size = len(files)

    def __len__(self):
        """Returns the number of items in the dataset"""
        return self.dataset_size

    def __getitem__(self, item):
        """Returns the item of the dataset"""


class TrainDataset(BaseDataset):
    """Train dataset for HiFaceGAN model"""

    def __init__(self, data_path, img_size):
        super().__init__(data_path)
        self.img_size = img_size

    def __getitem__(self, index):
        """Returns the item of the dataset"""
        if index >= self.dataset_size:
            raise IndexError(f'Index should be less than dataset size = {self.dataset_size}')

        file = self.img_path[index]
        org = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        if random.random() > 0.5:
            org = org[:, ::-1]
        h, w = org.shape[:2]
        sz = min(w, h)
        if h > w:
            dh = sz
            img_data = [org[dh * i:dh * (i + 1), 0:w] for i in range(h // dh)]
        else:
            dw = sz
            img_data = [org[0:h, dw * i:dw * (i + 1)] for i in range(w // dw)]

        lq = cv2.resize(img_data[0], (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        hq = cv2.resize(img_data[1], (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        lq = lq.transpose(2, 0, 1) / 255.
        hq = hq.transpose(2, 0, 1) / 255.

        return ms.Tensor(lq, dtype=ms.float32), ms.Tensor(hq, dtype=ms.float32)


class TestDataset(BaseDataset):
    """Test dataset for HiFaceGAN model"""

    def __init__(self, data_path, img_size):
        super().__init__(data_path)
        self.img_size = img_size

    def __getitem__(self, index):
        """Returns the item of the dataset"""
        if index >= self.dataset_size:
            raise IndexError(f'Index should be less than dataset size = {self.dataset_size}')

        path = self.img_path[index]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) / 255.

        lq_image_tensor = ms.Tensor(image[:, :self.img_size], dtype=ms.float32)
        hq_image_tensor = ms.Tensor(image[:, self.img_size:], dtype=ms.float32)
        return lq_image_tensor, hq_image_tensor


def create_train_dataset(data_root, degradation_type, batch_size, is_distributed,
                         group_size=1, rank=0, img_size=512):
    """Create train dataset for HiFaceGAN model"""
    data_path = Path(data_root) / f'train_{degradation_type}'
    dataset = TrainDataset(data_path, img_size)
    dataset_size = len(dataset) // group_size // batch_size
    if is_distributed:
        dataset = ds.GeneratorDataset(dataset, ['low_quality', 'high_quality'], shuffle=True,
                                      num_shards=group_size, shard_id=rank)
    else:
        dataset = ds.GeneratorDataset(dataset, ['low_quality', 'high_quality'], shuffle=True)

    return dataset.batch(batch_size, drop_remainder=True), dataset_size


def create_eval_dataset(data_root, degradation_type, batch_size, img_size=512):
    """Create eval dataset for HiFaceGAN model"""
    data_path = Path(data_root) / f'test_{degradation_type}'
    dataset = TestDataset(data_path, img_size)
    dataset_size = len(dataset) // batch_size
    dataset = ds.GeneratorDataset(dataset, ['low_quality', 'high_quality'], shuffle=False)
    return dataset.batch(batch_size, drop_remainder=True), dataset_size
