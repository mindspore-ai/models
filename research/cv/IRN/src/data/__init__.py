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
Helpers for creating SRDataset.
"""

import os

from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.dataset as ds

from .dataset import SRDataset


class Sampler():
    """Sampler distributes samples to workers evenly."""

    def __init__(self, num_data, rank_id, rank_size):
        self._num_data = num_data
        self._rank_id = rank_id
        self._rank_size = rank_size
        self._samples = int(num_data / rank_size)
        self._total_samples = self._samples * rank_size

    def __iter__(self):
        begin = self._rank_id * self._samples
        end = begin + self._samples
        indices = range(begin, end)
        return iter(indices)

    def __len__(self):
        return self._samples


def _get_rank_info():
    """Get rank size and rank id."""
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def create_dataset(dataset_path, scale, do_train=True, repeat_num=1,
                   batch_size=8, target="Ascend", distribute=False):
    """
    Create an SRDataset for training or testing.

    Args:
        dataset_path (string): Path to the dataset.
        scale (int): downscaling ratio.
        do_train (bool): Whether dataset is used for training or testing.
        repeat_num (int): Repeat times of the dataset.
        batch_size (int): Batch size of the dataset.
        target (str): Device target.
        distribute (bool): For distributed training or not.

    Returns:
        dataset
    """
    paths = []
    for p, _, fs in sorted(os.walk(dataset_path)):
        for f in sorted(fs):
            if f.endswith(".png"):
                paths.append(os.path.join(p, f))
    assert paths, "no png images found"
    sr_ds = SRDataset(paths, scale=scale, training=do_train)

    if target == "Ascend":
        rank_size, rank_id = _get_rank_info()
    else:
        if distribute:
            rank_id = get_rank()
            rank_size = get_group_size()
            sampler = Sampler(len(sr_ds), rank_id, rank_size)
        else:
            sampler = None
            rank_size = 1
            rank_id = 0

    num_shards = None if rank_size == 1 else rank_size
    shard_id = None if rank_size == 1 else rank_id
    if do_train:
        dataset = ds.GeneratorDataset(
            sr_ds, ["downscaled", "original"],
            num_parallel_workers=4 * rank_size, shuffle=True,
            sampler=sampler,
            num_shards=num_shards, shard_id=shard_id,
        )
    else:
        dataset = ds.GeneratorDataset(
            sr_ds, ["downscaled", "original"],
            num_parallel_workers=1, shuffle=False
        )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_num)
    return dataset
