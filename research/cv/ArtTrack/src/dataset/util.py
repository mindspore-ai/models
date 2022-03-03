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

import psutil
from mindspore import dataset as ds

DATASET_TYPE_MPII_RAW = 'mpii_raw'
DATASET_TYPE_MPII_MR = 'mpii_mr'
DATASET_TYPE_COCO = 'coco'


def _parallel(num):
    """
    get parallel
    """
    if num is None or num < 0:
        return max(1, int(psutil.cpu_count() / 2))
    return num


def create_dataset(dataset_type, dataset, shuffle=False, batch_size=1, parallel=None, train=False, num_shards=None,
                   rank_id=None):
    """
    create dataset
    Args:
        dataset_type: dataset type
        dataset: path to dataset
        shuffle: shuffle
        batch_size: batch size
        parallel: if None, cpu_count / 2
        train: train mode
        num_shards: shards for distributed training
        rank_id: distributed id
    """
    ds.config.set_enable_shared_mem(False)
    if dataset_type == DATASET_TYPE_MPII_RAW:
        columns = [
            "inputs",
            "part_score_targets",
            "part_score_weights",
            "locref_targets",
            "locref_mask",
        ]
        if not train:
            columns = ['im_path'] + columns
        dataset = ds.GeneratorDataset(source=dataset, column_names=columns, shuffle=shuffle,
                                      num_parallel_workers=_parallel(parallel), num_shards=num_shards,
                                      shard_id=rank_id)
    elif dataset_type == DATASET_TYPE_COCO:
        columns = [
            "inputs",
            "part_score_targets",
            "part_score_weights",
            "locref_targets",
            "locref_mask",
            "pairwise_targets",
            "pairwise_mask",
        ]
        if not train:
            columns = ['im_path'] + columns
        dataset = ds.GeneratorDataset(source=dataset, column_names=columns, shuffle=shuffle,
                                      num_parallel_workers=_parallel(parallel), num_shards=num_shards,
                                      shard_id=rank_id)
    if dataset is not None:
        dataset = dataset.batch(batch_size)

    return dataset
