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
import os

from mindspore import dataset as ds
from mindspore.communication.management import get_rank, get_group_size


def get_dataLoader(source, opts, args, column_names):
    if args.distributed:
        rank_id = get_rank()
        rank_size = get_group_size()
    if isinstance(source, tuple):
        data_loaders_pos = []
        data_loaders_neg = []
        datasets_pos, datasets_neg = source
        if not args.distributed:
            for dataset_pos in datasets_pos:
                dataset = ds.GeneratorDataset(source=dataset_pos,
                                              column_names=column_names,
                                              num_parallel_workers=args.num_workers, shuffle=True)
                dataset = dataset.batch(batch_size=opts['minibatch_size'])
                data_loaders_pos.append(dataset)
            for dataset_neg in datasets_neg:
                dataset = ds.GeneratorDataset(source=dataset_neg,
                                              column_names=column_names,
                                              num_parallel_workers=args.num_workers, shuffle=True)
                dataset = dataset.batch(batch_size=opts['minibatch_size'])
                data_loaders_neg.append(dataset)
        else:
            for dataset_pos in datasets_pos:
                dataset = ds.GeneratorDataset(source=dataset_pos,
                                              column_names=column_names,
                                              num_parallel_workers=args.num_workers, shuffle=True, num_shards=rank_size,
                                              shard_id=rank_id)
                dataset = dataset.batch(batch_size=opts['minibatch_size'])
                data_loaders_pos.append(dataset)
            for dataset_neg in datasets_neg:
                dataset = ds.GeneratorDataset(source=dataset_neg,
                                              column_names=["im", "bbox", "action_label", "score_label", "vid_idx"],
                                              num_parallel_workers=args.num_workers, shuffle=True, num_shards=rank_size,
                                              shard_id=rank_id)
                dataset = dataset.batch(batch_size=opts['minibatch_size'])
                data_loaders_neg.append(dataset)
        return data_loaders_pos, data_loaders_neg
    if args.distributed:
        dataset = ds.GeneratorDataset(source=source,
                                      column_names=column_names,
                                      num_parallel_workers=args.num_workers, shuffle=True, num_shards=rank_size,
                                      shard_id=rank_id)
        dataset = dataset.batch(batch_size=opts['minibatch_size'])
    else:
        dataset = ds.GeneratorDataset(source=source,
                                      column_names=column_names,
                                      num_parallel_workers=args.num_workers, shuffle=True)
        dataset = dataset.batch(batch_size=opts['minibatch_size'])
    return dataset


def get_groundtruth(gt_path):
    if not os.path.exists(gt_path):
        bboxes = []
        t_sum = 0
        return bboxes, t_sum

    # parse gt
    gtFile = open(gt_path, 'r')
    gt = gtFile.read().split('\n')
    for i in range(len(gt)):
        if gt[i] == '' or gt[i] is None:
            continue
        if ',' in gt[i]:
            separator = ','
        elif '\t' in gt[i]:
            separator = '\t'
        elif ' ' in gt[i]:
            separator = ' '
        else:
            separator = ','

        gt[i] = gt[i].split(separator)
        gt[i] = list(map(float, gt[i]))
    gtFile.close()

    if len(gt[0]) >= 6:
        for gtidx in range(len(gt)):
            if gt[gtidx] == "":
                continue
            x = gt[gtidx][0:len(gt[gtidx]):2]
            y = gt[gtidx][1:len(gt[gtidx]):2]
            gt[gtidx] = [min(x), min(y), max(x) - min(x), max(y) - min(y)]

    return gt
