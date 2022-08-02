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
"""Data Preprocessing app."""

# This script should be run directly with 'python <script> <args>'.

import os
import io
import json
import math

from src.path_gen import PathGen
from src.config import TBNetConfig
from src.utils.param import param


def preprocess_csv(path_gen, data_home, src_name, out_name):
    """Pre-process a csv file."""
    src_path = os.path.join(data_home, src_name)
    out_path = os.path.join(data_home, out_name)
    print(f'converting {src_path} to {out_path} ...')
    rows = path_gen.generate(src_path, out_path)
    print(f'{rows} rows of path data generated.')


def preprocess_data():
    """Pre-process the dataset."""
    data_home = os.path.join(param.data_path, 'data', param.dataset)
    config_path = os.path.join(data_home, 'config.json')
    id_maps_path = os.path.join(data_home, 'id_maps.json')

    cfg = TBNetConfig(config_path)
    if param.device_target == 'Ascend':
        cfg.per_item_paths = math.ceil(cfg.per_item_paths / 16) * 16
    path_gen = PathGen(per_item_paths=cfg.per_item_paths, same_relation=param.same_relation)

    preprocess_csv(path_gen, data_home, 'src_train.csv', 'train.csv')

    # save id maps for the later use by Recommender in infer.py
    with io.open(id_maps_path, mode="w", encoding="utf-8") as f:
        json.dump(path_gen.id_maps(), f, indent=4)

    # count distinct objects from the training set
    cfg.num_items = path_gen.num_items
    cfg.num_references = path_gen.num_references
    cfg.num_relations = path_gen.num_relations
    cfg.save(config_path)

    print(f'{config_path} updated.')
    print(f'num_items: {cfg.num_items}')
    print(f'num_references: {cfg.num_references}')
    print(f'num_relations: {cfg.num_relations}')

    # treat new items and references in test and infer set as unseen entities
    # dummy internal id 0 will be assigned to them
    path_gen.grow_id_maps = False

    preprocess_csv(path_gen, data_home, 'src_test.csv', 'test.csv')

    # for inference, only take interacted('c') and other('x') items as candidate items,
    # the purchased('p') items won't be recommended.
    # assume there is only one user in src_infer.csv
    path_gen.subject_ratings = "cx"
    preprocess_csv(path_gen, data_home, 'src_infer.csv', 'infer.csv')

    print(f'Dataset {data_home} processed.')


if __name__ == '__main__':
    preprocess_data()
