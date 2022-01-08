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

"""arbitrary style transfer train dataset."""
import os
import math
import random
import numpy as np
from PIL import Image
from mindspore import context
import mindspore.dataset as ds
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank


class mydata:
    """Import dataset"""

    def __init__(self, opts, content_path, style_path):
        """init"""
        self.content_path = content_path
        self.style_path = style_path
        self.reshape_size = opts.reshape_size
        self.crop_size = opts.crop_size
        self.content_img = os.listdir(self.content_path)
        self.style_img = os.listdir(self.style_path)
        print('train content size:', len(self.content_img), 'train style size:', len(self.style_img))
        np.random.shuffle(self.content_img)
        np.random.shuffle(self.style_img)
        print('content:', self.content_path)
        print('style:', self.style_path)

    def __len__(self):
        """getlength"""
        return len(self.content_img)

    def __getitem__(self, i):
        """getitem"""
        # img_item={}
        content_index = i
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
        # augmentation
        hor_flip = random.randrange(0, 2)
        if hor_flip:
            img_c = np.fliplr(img_c)
            img_s = np.fliplr(img_s)
        img_c = img_c.transpose(2, 0, 1).astype(np.float32)
        img_s = img_s.transpose(2, 0, 1).astype(np.float32)
        return img_c, img_s


class MySampler():
    """sampler for distribution"""

    def __init__(self, dataset, local_rank, world_size):
        self.__num_data = len(dataset)
        self.__local_rank = local_rank
        self.__world_size = world_size
        self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
        self.total_num_samples = self.samples_per_rank * self.__world_size

    def __iter__(self):
        """"iter"""
        indices = list(range(self.__num_data))
        indices.extend(indices[:self.total_num_samples - len(indices)])
        indices = indices[self.__local_rank:self.total_num_samples:self.__world_size]
        return iter(indices)

    def __len__(self):
        """length"""
        return self.samples_per_rank


def create_traindataset(opts, content_path, style_path):
    """create arbitrary style transfer dataset"""
    parallel_mode = context.get_auto_parallel_context("parallel_mode")

    if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
        dataset = mydata(opts, content_path, style_path)
        device_num = opts.device_num
        rank_id = get_rank()
        sampler = MySampler(dataset, local_rank=rank_id, world_size=device_num)
        DS = ds.GeneratorDataset(dataset, column_names=['content', 'style'], shuffle=True,
                                 num_shards=device_num, shard_id=rank_id, sampler=sampler, num_parallel_workers=8)
        DS = DS.batch(opts.batchsize, drop_remainder=True)
    else:
        dataset = mydata(opts, content_path, style_path)
        DS = ds.GeneratorDataset(dataset, column_names=['content', 'style'], shuffle=True, num_parallel_workers=8)
        DS = DS.batch(opts.batchsize)
    return DS
