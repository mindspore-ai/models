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
import json
import numpy as np

import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size

from src.utils.common import  pc_normalize, random_scale_point_cloud, shift_point_cloud


class PartNormalDataset():
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.num_points = cfg.num_points

        data_path = os.path.join(cfg.dataset_path, 'shapenetcore_partanno_segmentation_benchmark_v0_normal')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path {data_path} not found")

        self.normal_channel = cfg.use_normals

        self.catfile = os.path.join(data_path, 'synsetoffset2category.txt')
        self.category = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.category[ls[0]] = ls[1]
        self.category = {key: value for key, value in self.category.items()}
        self.classes_original = dict(zip(self.category, range(len(self.category))))

        self.meta = {}
        for it in self.category:
            self.meta[it] = []
            pointdir = os.path.join(data_path, self.category[it])
            file_names = sorted(os.listdir(pointdir))
            json_path = os.path.join(data_path, 'train_test_split')
            if mode == 'trainval':
                with open(os.path.join(json_path, 'shuffled_train_file_list.json'), 'r') as f:
                    train_idx = {[str(d.split('/')[2]) for d in json.load(f)]}
                with open(os.path.join(json_path, 'shuffled_val_file_list.json'), 'r') as f:
                    val_idx = {str(d.split('/')[2]) for d in json.load(f)}
                file_names = [fn for fn in file_names if ((fn[0:-4] in train_idx) or (fn[0:-4] in val_idx))]
            else:
                with open(os.path.join(json_path, f'shuffled_{mode}_file_list.json'), 'r') as f:
                    idx = {str(d.split('/')[2]) for d in json.load(f)}
                file_names = [fn for fn in file_names if fn[0:-4] in idx]

            for fn in file_names:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[it].append(os.path.join(pointdir, token + '.txt'))

        self.datapath = []
        for it in self.category:
            for fn in self.meta[it]:
                self.datapath.append((it, fn))

        self.classes = {}
        for it in self.category:
            self.classes[it] = self.classes_original[it]

        self.cache = {}
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point = data[:, 0:3]
            else:
                point = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point, cls, seg)
        point[:, 0:3] = pc_normalize(point[:, 0:3])

        choice = np.random.choice(len(seg), self.num_points, replace=True)
        # resample
        point = point[choice, :]
        seg = seg[choice]

        return point, cls, seg

    def __len__(self):
        return len(self.datapath)

def create_shapenet_dataset(mode, cfg):
    dataset_generator = PartNormalDataset(cfg, mode=mode)
    num_workers = cfg.num_workers
    batch_size = cfg.batch_size
    is_shuffe = False
    if mode == 'train':
        is_shuffe = True

    if (cfg.run_distribute) and (mode == 'train'):
        num_shards = get_group_size()
        shard_id = get_rank()
        sampler = ds.DistributedSampler(num_shards, shard_id, is_shuffe)
        dataset = ds.GeneratorDataset(dataset_generator,
                                      ["point_set", "cls", "seg"],
                                      sampler=sampler,
                                      num_parallel_workers=num_workers)
    else:
        dataset = ds.GeneratorDataset(dataset_generator, ["point_set", "cls", "seg"],
                                      shuffle=is_shuffe, num_parallel_workers=num_workers)
    if mode == 'train':
        trans = [random_scale_point_cloud(), shift_point_cloud()]
        dataset = dataset.map(operations=trans,
                              input_columns="point_set",
                              num_parallel_workers=num_workers)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset
