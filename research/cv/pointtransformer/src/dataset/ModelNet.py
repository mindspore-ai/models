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
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size

from src.utils.common import farthest_point_sample, pc_normalize, random_point_dropout, random_scale_point_cloud, shift_point_cloud


class ModelNetDataset():
    def __init__(self, mode, cfg):
        super().__init__()

        self.npoints = cfg.num_points
        self.uniform = cfg.uniform
        self.use_normals = cfg.use_normals

        data_path = os.path.join(cfg.dataset_path, 'modelnet40_normal_resampled')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path {data_path} not found")

        self.catfile = os.path.join(data_path, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids[mode] = [line.rstrip() for line in open(os.path.join(data_path, f'modelnet40_{mode}.txt'))]

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[mode]]
        self.data_list = [(shape_names[i], os.path.join(data_path, shape_names[i], shape_ids[mode][i]) + '.txt')
                          for i in range(len(shape_ids[mode]))]


    def __getitem__(self, index):
        fn = self.data_list[index]
        cls = self.classes[self.data_list[index][0]]
        label = np.array([cls]).astype(np.int32)
        point = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        if self.uniform:
            point = farthest_point_sample(point, self.npoints)
        else:
            point = point[0: self.npoints, :]

        point[:, 0:3] = pc_normalize(point[:, 0:3])
        if not self.use_normals:
            point = point[:, 0:3]

        return point, label[0]

    def __len__(self):
        return len(self.data_list)


def create_modelnet40_dataset(mode, cfg):
    dataset_generator = ModelNetDataset(mode, cfg)
    num_workers = cfg.num_workers
    batch_size = cfg.batch_size

    if cfg.run_distribute and mode == 'train':
        num_shards = get_group_size()
        shard_id = get_rank()
        sampler = ds.DistributedSampler(num_shards, shard_id, shuffle=True)
        dataset = ds.GeneratorDataset(dataset_generator,
                                      ["point_set", "cls"],
                                      sampler=sampler,
                                      num_parallel_workers=num_workers)
    elif mode == 'train':
        dataset = ds.GeneratorDataset(dataset_generator,
                                      ["point_set", "cls"],
                                      shuffle=True,
                                      num_parallel_workers=num_workers)
        trans = [random_point_dropout(), random_scale_point_cloud(), shift_point_cloud()]
        dataset = dataset.map(operations=trans,
                              input_columns="point_set",
                              num_parallel_workers=num_workers)
    else:
        dataset = ds.GeneratorDataset(dataset_generator,
                                      ["point_set", "cls"],
                                      shuffle=False,
                                      num_parallel_workers=num_workers)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset
