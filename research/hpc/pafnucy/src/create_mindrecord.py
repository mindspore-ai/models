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
"""create mindrecord"""

import os.path
import pandas as pd
import numpy as np
from mindspore.mindrecord import FileWriter
from src.model_utils.config import config
from src.dataloader import preprocess_dataset


class DatasetIter:
    def __init__(self, coor_features, affinity):
        self.coor_features = coor_features
        self.affinity = affinity

    def __getitem__(self, index):
        return self.coor_features[index], self.affinity[index]

    def __len__(self):
        return len(self.coor_features)


def create_mindrecord():
    train_coords_features, y_train_size, train_no_rotationcoords_features, \
    val_coords_features, no_batch_size = preprocess_dataset(config, config.data_path,
                                                            batch_rotation=list(range(config.rotations)),
                                                            batch_no_rotation=0, v_batch_rotation=0)
    print("train size: ", y_train_size, flush=True)
    print("Validation size: ", no_batch_size, flush=True)
    train_rotation_path = os.path.join(config.mindrecord_path, 'train_rotation')
    train_no_rotation_path = os.path.join(config.mindrecord_path, 'no_rotation')
    val_path = os.path.join(config.mindrecord_path, 'val')
    if not os.path.exists(train_rotation_path):
        os.mkdir(train_rotation_path)
    if not os.path.exists(train_no_rotation_path):
        os.mkdir(train_no_rotation_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    train_rot_writer = FileWriter(os.path.join(train_rotation_path, 'train_rotation_dataset.mindrecord'), shard_num=1)
    rot_train_data_schema = {
        "coords_features": {"type": "float32", "shape": [19, 21, 21, 21]},
        "affinitys": {"type": "float32", "shape": [-1]}
    }
    train_rot_writer.add_schema(rot_train_data_schema, "pdbbind_rot")
    data_iterator = DatasetIter(train_coords_features['coords_features'], train_coords_features['affinitys'])
    train_rot_item = {'coords_features': [], 'affinitys': []}
    for coor_feature, affine in data_iterator:
        train_rot_item['coords_features'] = np.array(coor_feature, dtype=np.float32)
        train_rot_item['affinitys'] = np.array(affine, dtype=np.float32)
        train_rot_writer.write_raw_data([train_rot_item])
    train_rot_writer.commit()
    print("Rotation training mindrecord create finished!", flush=True)

    train_norot_writer = FileWriter(os.path.join(train_no_rotation_path,
                                                 'train_norotation_dataset.mindrecord'), shard_num=1)
    norot_train_data_schema = {
        "coords_features": {"type": "float32", "shape": [19, 21, 21, 21]},
        "affinitys": {"type": "float32", "shape": [-1]}
    }
    train_norot_writer.add_schema(norot_train_data_schema, "pdbbind_norot")
    norot_data_iterator = DatasetIter(train_no_rotationcoords_features['coords_features'],
                                      train_no_rotationcoords_features['affinitys'])
    train_no_rot_item = {'coords_features': [], 'affinitys': []}
    for coor_feature, affine in norot_data_iterator:
        train_no_rot_item['coords_features'] = np.array(coor_feature, dtype=np.float32)
        train_no_rot_item['affinitys'] = np.array(affine, dtype=np.float32)
        train_norot_writer.write_raw_data([train_no_rot_item])
    train_norot_writer.commit()
    print("No rotation training mindrecord create finished!", flush=True)

    val_writer = FileWriter(os.path.join(val_path, 'validation_dataset.mindrecord'), shard_num=1)
    val_data_schema = {
        "coords_features": {"type": "float32", "shape": [19, 21, 21, 21]},
        "affinitys": {"type": "float32", "shape": [-1]}
    }
    val_writer.add_schema(val_data_schema, "pdbbind_val")
    val_data_iterator = DatasetIter(val_coords_features['coords_features'],
                                    val_coords_features['affinitys'])
    val_rot_item = {'coords_features': [], 'affinitys': []}
    for coor_feature, affine in val_data_iterator:
        val_rot_item['coords_features'] = np.array(coor_feature, dtype=np.float32)
        val_rot_item['affinitys'] = np.array(affine, dtype=np.float32)
        val_writer.write_raw_data([val_rot_item])
    val_writer.commit()
    size_list = [{'dataset': 'train_size', "size": y_train_size},
                 {'dataset': 'val_size', "size": no_batch_size}]
    results = pd.DataFrame(size_list, columns=['dataset', 'size'])
    results.to_csv(os.path.join(config.mindrecord_path, 'ds_size.csv'), index=False)
    print("Validation mindrecord create finished!", flush=True)


if __name__ == '__main__':
    create_mindrecord()
