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

import os

import numpy as np
import scipy.io

from src.dataset.coco import MSCOCO
from src.dataset.pose import Batch
from src.tool.decorator import process_cfg


def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]


def save_stats(stats, cfg):
    mat_stats = {"graph": [], "means": [], "std_devs": []}
    for start in range(cfg.num_joints):
        for end in range(cfg.num_joints):
            if start != end:
                joint_pair = (start, end)
                mat_stats["graph"].append([start, end])
                mat_stats["means"].append(stats[joint_pair]["mean"])
                mat_stats["std_devs"].append(stats[joint_pair]["std"])
    print(mat_stats)
    os.makedirs(os.path.dirname(cfg.pairwise_stats_fn), exist_ok=True)
    scipy.io.savemat(cfg.pairwise_stats_fn, mat_stats)


# Compute pairwise statistics at reference scale
@process_cfg
def pairwise_stats(cfg):
    dataset = MSCOCO(cfg)
    dataset.set_pairwise_stats_collect(True)

    num_images = dataset.num_images
    all_pairwise_differences = {}

    if cfg.dataset.mirror:
        num_images *= 2

    for k in range(num_images):
        print('processing image {}/{}'.format(k, num_images - 1))

        batch = dataset.get_item(k)
        batch_stats = batch[Batch.data_item].pairwise_stats
        for joint_pair in batch_stats:
            if joint_pair not in all_pairwise_differences:
                all_pairwise_differences[joint_pair] = []
            all_pairwise_differences[joint_pair] += batch_stats[joint_pair]

    stats = {}
    for joint_pair in all_pairwise_differences:
        stats[joint_pair] = {}
        stats[joint_pair]["mean"] = np.mean(all_pairwise_differences[joint_pair], axis=0)
        stats[joint_pair]["std"] = np.std(all_pairwise_differences[joint_pair], axis=0)

    save_stats(stats, cfg)
