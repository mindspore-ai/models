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
from copy import deepcopy


class ReIDDistributedSampler:
    """ Distributed sampler for ReID

    Args:
        subset: dataset
        batch_id: number of ids in batch
        batch_image: number of id images in batch
        rank: process id
        group_size: device number
        seed: inner random state seed
    """

    def __init__(self, subset, batch_id, batch_image, rank=0, group_size=1, seed=7225):

        self.subset = subset
        self.batch_image = batch_image
        self.batch_id = batch_id
        self.rank = rank
        self.group_size = group_size

        self.rng = random.Random(seed)

        self._id2index = collections.defaultdict(list)

        for idx, (_, pid, _) in enumerate(subset):
            self._id2index[pid].append(idx)

        self._unique_ids = []
        for pid, idxs in self._id2index.items():
            for st in range(0, len(idxs) - self.batch_image + 1, self.batch_image):
                self._unique_ids.append((pid, st))

        self._num_group_ids = len(self._unique_ids) // self.group_size
        self._shift_id = self.rank * self._num_group_ids

    def _shuffle_img(self, id2index):
        """ Shuffle images for every pid """
        id2index = deepcopy(id2index)
        for pid in id2index:
            self.rng.shuffle(id2index[pid])
        return id2index

    def __iter__(self):
        """ Get image iterator """
        id2index = self._shuffle_img(self._id2index)

        unique_ids = self._unique_ids.copy()
        self.rng.shuffle(unique_ids)

        imgs = []

        unique_ids = unique_ids[self._shift_id:self._shift_id + self._num_group_ids]
        for pid, st in unique_ids:
            imgs.extend(id2index[pid][st:st + self.batch_image])

        return iter(imgs)

    def __len__(self):
        """ Sampler length """
        return self._num_group_ids * self.batch_image
