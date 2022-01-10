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
Data operations, will be used in train.py and eval.py
"""
import numpy as np
import mindspore.dataset as ds

def get_data_set(word_index, tag_index, batch_size):
    """get the data for train and eval"""
    def generator_func():
        for i in range(len(word_index)):
            yield (np.array([j for j in word_index[i]]).astype(np.int32),
                   np.array([value for value in tag_index[i]]).astype(np.int32))

    data_set = ds.GeneratorDataset(generator_func, ["feature", "label"])
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
