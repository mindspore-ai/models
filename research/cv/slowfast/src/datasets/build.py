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
"""Build dataset"""
import mindspore.dataset as ds
from src.datasets.ava_dataset import Ava

ds.config.set_prefetch_size(8)
ds.config.set_numa_enable(True)

def build_dataset(cfg, split, num_shards=None, shard_id=None, device_target='Ascend'):
    """
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset.
    """
    assert split in ["train", "test"], 'split must be in train or test'

    dataset_generator = Ava(cfg, split)

    if split == 'train':
        dataset = ds.GeneratorDataset(dataset_generator,
                                      ["slowpath", "fastpath", "boxes", "labels", "mask"],
                                      num_parallel_workers=16 if device_target == 'Ascend' else 6,
                                      python_multiprocessing=False,
                                      shuffle=True,
                                      num_shards=num_shards,
                                      shard_id=shard_id)
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE)
    else:
        dataset = ds.GeneratorDataset(dataset_generator,
                                      ["slowpath", "fastpath", "boxes", "labels", "ori_boxes", "metadata", "mask"],
                                      num_parallel_workers=16 if device_target == 'Ascend' else 6,
                                      python_multiprocessing=False,
                                      shuffle=False)
        dataset = dataset.batch(cfg.TEST.BATCH_SIZE)

    if dataset.get_dataset_size() == 0:
        raise ValueError("dataset size is 0, please check dataset size > 0 and batch_size <= dataset size")
    return dataset
