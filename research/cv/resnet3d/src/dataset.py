# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
create dataset.
"""
import os

import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms as C2
from mindspore.communication.management import get_rank, get_group_size

from .videodataset import DatasetGenerator
from .videodataset_multiclips import DatasetGeneratorMultiClips
from .temporal_transforms import (TemporalRandomCrop, TemporalCenterCrop,
                                  SlidingWindow, TemporalSubsampling)
from .temporal_transforms import Compose as TemporalCompose
from .pil_transforms import PILTrans, EvalPILTrans


def create_train_dataset(root_path, annotation_path, opt, repeat_num=1, batch_size=32, target="Ascend"):
    """
    create_eval_dataset.
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_dataset = DatasetGenerator(root_path=root_path, annotation_path=annotation_path,
                                     subset='training', temporal_transform=temporal_transform)

    if device_num == 1:
        dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"],
                                      num_parallel_workers=4, shuffle=True)
    else:
        dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"],
                                      num_parallel_workers=4, shuffle=True, num_shards=device_num,
                                      shard_id=rank_id)
    type_cast_op = C2.TypeCast(mstype.int32)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    trans = PILTrans(opt, mean=mean, std=std)
    dataset = dataset.map(operations=type_cast_op,
                          input_columns='label', num_parallel_workers=4)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True, per_batch_map=trans, num_parallel_workers=4,
                            input_columns=['data', 'label'])
    dataset = dataset.repeat(repeat_num)
    return dataset


def create_eval_dataset(root_path, annotation_path, opt, target="Ascend"):
    """
    create_eval_dataset.
    """
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)
    eval_dataset = DatasetGeneratorMultiClips(root_path=root_path, annotation_path=annotation_path,
                                              subset='validation', temporal_transform=temporal_transform,
                                              target_type=['video_id', 'segment'])

    dataset = ds.GeneratorDataset(eval_dataset, column_names=["data", "label"],
                                  num_parallel_workers=4, shuffle=False)

    type_cast_op = C2.TypeCast(mstype.int32)
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]

    trans = EvalPILTrans(opt, mean=mean, std=std)
    dataset = dataset.map(operations=type_cast_op, input_columns='label')
    dataset = dataset.batch(batch_size=1, drop_remainder=True, per_batch_map=trans, input_columns=['data', 'label'],
                            num_parallel_workers=4)
    return dataset


def _get_rank_info():
    """
    get rank size and rank id.
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0
    return rank_size, rank_id
