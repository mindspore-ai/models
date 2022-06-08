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
""" dataset """
import numpy as np
import cv2
import mindspore.dataset.vision as C
import mindspore.dataset as ds
from mindspore.common import set_seed
cv2.setNumThreads(0)
set_seed(1)


class SegDataset:
    """init dataset"""
    def __init__(self,
                 image_mean,
                 image_std,
                 data_file='',
                 batch_size=32,
                 crop_size=512,
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21,
                 num_readers=2,
                 num_parallel_calls=4,
                 shard_id=None,
                 shard_num=None):
        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        self.enable_flip = True
        assert max_scale > min_scale

    def expand(self, image, label):
        """expand image"""
        if np.random.uniform(0.0, 1.0) > 0.5:
            return image, label
        h, w, c = image.shape
        ratio = np.random.uniform(1.0, 4.0)
        mean = (0, 0, 0)
        expand_img = np.full((int(h * ratio), int(w * ratio), c), mean).astype(image.dtype)
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = image
        image = expand_img
        expand_label = np.full((int(h * ratio), int(w * ratio)), self.ignore_label).astype(label.dtype)
        expand_label[top:top + h, left:left + w] = label
        label = expand_label
        return image, label

    def resize_long(self, img):
        """resize"""
        long_size = self.crop_size
        h, w, _ = img.shape
        if h > w:
            new_h = long_size
            new_w = int(1.0 * long_size * w / h)
        else:
            new_w = long_size
            new_h = int(1.0 * long_size * h / w)
        return new_h, new_w

    def RandomScaleAndCrop(self, image, label):
        """random scale and crop"""
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image.shape[0]), int(sc * image.shape[1])
        image_out = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        image_out = (image_out - self.image_mean) / self.image_std
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w+self.crop_size]
        return image_out, label_out

    def preprocess_(self, image, label):
        """bgr image"""
        image_out = image
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        image_out, label_out = self.RandomScaleAndCrop(image_out, label_out)
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]
        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def get_dataset1(self):
        """get dataset"""
        ds.config.set_seed(1000)
        data_set = ds.MindDataset(self.data_file, columns_list=["data", "label"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id)
        decode_op = C.Decode()
        trans = [decode_op]
        data_set = data_set.map(operations=trans, input_columns=["data"])
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["data", "label"],
                                output_columns=["data", "label"],
                                num_parallel_workers=self.num_parallel_calls)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        return data_set
