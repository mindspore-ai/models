# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Data operations, will be used in train.py."""

import mindspore.dataset as de

de.config.set_seed(1)
de.config.set_prefetch_size(96)

def fun(data, shape):
    data = data.reshape(shape)
    return data[0], data[1], data[2], data[3], data[4], data[5]


def create_transformer_dynamic_dataset(dataset_path=None, rank_size=1, rank_id=0, do_shuffle="true"):
    """
    create transformer dynamic dataset.
    """
    dataset = de.MindDataset(dataset_path,
                             columns_list=["batch_data", "batch_shape"],
                             shuffle=(do_shuffle == "true"), num_shards=rank_size, shard_id=rank_id)

    dataset = dataset.map(fun, input_columns=["batch_data", "batch_shape"],
                          output_columns=["source_eos_ids", "source_eos_mask",
                                          "target_sos_ids", "target_sos_mask",
                                          "target_eos_ids", "target_eos_mask"],
                          )
    dataset = dataset.project(["source_eos_ids", "source_eos_mask",
                               "target_sos_ids", "target_sos_mask",
                               "target_eos_ids", "target_eos_mask"])
    return dataset
