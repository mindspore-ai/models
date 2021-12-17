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
""" Choose samples from the dataset """
import collections
import random


class ReIDDistributedSampler:
    """ Distributed sampler for ReID

    Args:
        data_source: dataset
        batch_id: number of ids in batch
        batch_image: number of id images in batch
        rank: process id
        group_size: device number
        seed: inner random state seed
    """
    def __init__(self, data_source, batch_id, batch_image, rank=0, group_size=1, seed=7225):

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id
        self.rank = rank
        self.group_size = group_size

        self.rng = random.Random(seed)

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)

        self._shift = self.rank * len(self)

    def _sample(self, population, k):
        """ Get k samples of population """
        if len(population) < k:
            population = population * k
        return self.rng.sample(population, k)

    def __iter__(self):
        """ Get image iterator """
        unique_ids = self.data_source.unique_ids.copy()
        self.rng.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))

        imgs = imgs[self._shift:self._shift+len(self)]

        return iter(imgs)

    def __len__(self):
        """ Sampler length """
        return (len(self._id2index) // self.group_size) * self.batch_image
