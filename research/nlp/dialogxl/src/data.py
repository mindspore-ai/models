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

import pickle
import mindspore as ms
from mindspore import Tensor

class Dataset:
    def __init__(self, content_ids, labels, content_mask, content_lengths, speaker_ids):
        self.content_ids = content_ids
        self.labels = labels
        self.content_mask = content_mask
        self.content_lengths = content_lengths
        self.speaker_ids = speaker_ids

    def __getitem__(self, index):
        return Tensor(self.content_ids[index], dtype=ms.int32), \
               Tensor(self.labels[index], dtype=ms.int32), \
               Tensor(self.content_mask[index], dtype=ms.float32), \
               self.content_lengths[index], \
               Tensor(self.speaker_ids[index], dtype=ms.float32)

    def __len__(self):
        return len(self.content_ids)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    datasets = [Dataset(*d) for d in data]
    return datasets
