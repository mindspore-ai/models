"""construct the sampler"""
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

from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import mindspore.dataset as ds

class RandomIdentitySampler(ds.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, rank_id, rank_size, num_instances=4):
        super(RandomIdentitySampler, self).__init__()
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.__local_rank = rank_id
        self.__world_size = rank_size
        self.batch_size = 32
        self.pids_per_batchsize = self.batch_size//self.num_instances
        self.pids_per_npu = self.num_identities//self.__world_size
        self.samples_per_npu = self.pids_per_npu*self.num_instances

    def __iter__(self):
        indices = []
        for i in range(self.num_identities):
            indices.append(i)
        np.random.shuffle(indices)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = True
            if len(t) >= self.num_instances:
                replace = False
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        if self.__world_size != 1:
            ret2 = ret
            ret = ret2[self.__local_rank*self.samples_per_npu:(self.__local_rank+1)*self.samples_per_npu]
        return iter(ret)

    def __len__(self):
        return self.samples_per_npu
