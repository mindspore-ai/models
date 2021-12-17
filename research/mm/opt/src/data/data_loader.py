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

""" dataloader """

import os


class DataLoader:
    """ DataLoader """

    def __init__(self, dataset, batch_sampler, collate_fn, device_num=256):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collat_fn = collate_fn
        self.device_num = device_num
        rank_id_str = os.getenv('RANK_ID', '0')
        self.rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])

    def __iter__(self):
        self.step_index = 0
        self.batch_indices = iter(self.batch_sampler)
        return self

    def __next__(self):
        try:
            indices = next(self.batch_indices)
        except StopIteration:
            raise StopIteration
        data = []
        per_batch = len(indices) // self.device_num
        index = indices[self.rank_id * per_batch:(self.rank_id + 1) * per_batch]
        for idx in index:
            data.append(self.dataset[idx])

        data = self.collat_fn(data)
        return data
