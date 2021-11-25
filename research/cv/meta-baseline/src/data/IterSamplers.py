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
"""
CategoriesSampler
"""
import mindspore.dataset as ds
import numpy as np


class CategoriesSampler(ds.Sampler):
    """
    CategoriesSampler
    """

    def __init__(self, data, label, n_cls, n_per, iterations, ep_per_batch=1):
        super(CategoriesSampler, self).__init__()
        self.__iterations = iterations
        self.n_cls = n_cls  # way
        self.n_per = n_per  # shot = support_shot + query_shot
        self.ep_per_batch = ep_per_batch  # 4
        self.__iter = 0
        label = np.array(label)
        self.data = data
        self.label = label
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __next__(self):

        if self.__iter >= self.__iterations:
            raise StopIteration
        batch = []
        for _ in range(self.ep_per_batch):
            episode = []
            classes = np.random.choice(len(self.catlocs), self.n_cls,
                                       replace=False)
            for c in classes:
                l = np.random.choice(self.catlocs[c], self.n_per,
                                     replace=False)
                episode.append(self.data[l])
            episode = np.stack(episode)
            batch.append(episode)
        batch = np.stack(batch)  # bs * n_cls * n_per
        self.__iter += 1
        return (batch,)

    def __iter__(self):
        self.__iter = 0
        return self

    def __len__(self):
        return self.__iterations
