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

"""arbitrary style transfer test dataset."""
import os
import random
import numpy as np
from PIL import Image
import mindspore.dataset as ds


class mydata:
    """Import dataset"""

    def __init__(self, opts):
        """init"""
        self.content_path = opts.content_path
        self.style_path = opts.style_path
        self.reshape_size = opts.reshape_size
        self.crop_size = opts.crop_size
        self.content_img = os.listdir(self.content_path)
        self.style_img = os.listdir(self.style_path)
        print('test content size:', len(self.content_img), 'test style size:', len(self.style_img))
        sorted(self.content_img)
        sorted(self.style_img)

    def __len__(self):
        """getlength"""
        return len(self.content_img) * len(self.style_img)

    def __getitem__(self, i):
        """getitem"""
        content_index = i // (len(self.style_img))
        style_index = i % (len(self.style_img))
        img_c = Image.open(os.path.join(self.content_path, self.content_img[content_index])).convert("RGB")
        img_s = Image.open(os.path.join(self.style_path, self.style_img[style_index])).convert("RGB")

        # resize
        img_c = np.array(img_c.resize((self.reshape_size, self.reshape_size)))
        img_s = np.array(img_s.resize((self.reshape_size, self.reshape_size)))

        img_c = (img_c / 127.5) - 1.0
        img_s = (img_s / 127.5) - 1.0
        # crop
        ix = random.randrange(0, self.reshape_size - self.crop_size)
        iy = random.randrange(0, self.reshape_size - self.crop_size)
        img_c = img_c[iy: iy + self.crop_size, ix: ix + self.crop_size]
        img_s = img_s[iy: iy + self.crop_size, ix: ix + self.crop_size]
        img_c = img_c.transpose(2, 0, 1).astype(np.float32)
        img_s = img_s.transpose(2, 0, 1).astype(np.float32)
        return img_c, img_s


def create_testdataset(opts):
    """create arbitrary style transfer dataset"""
    dataset = mydata(opts)
    DS = ds.GeneratorDataset(dataset, column_names=['content', 'style'])
    DS = DS.batch(opts.batchsize)
    return DS
