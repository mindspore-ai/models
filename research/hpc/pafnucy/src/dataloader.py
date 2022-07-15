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
"""dataset loader"""

import os
import numpy as np
import h5py
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
from src.data import Featurizer, make_grid, rotate


def get_batch(configs, dataset_name, indices, coords, features, columns, std, affin, rotations=0):
    train_x_final = {'coords_features': [], 'affinitys': []}
    for rotation in rotations:
        for i, idx in enumerate(indices):
            x = []
            coords_idx = rotate(coords[dataset_name][idx], rotation)
            features_idx = features[dataset_name][idx]

            x.append(make_grid(coords_idx, features_idx,
                               grid_resolution=configs.grid_spacing,
                               max_dist=configs.max_dist))
            x = np.vstack(x)
            x[..., columns['partialcharge']] /= std
            train_x_final['coords_features'].append(np.transpose(np.squeeze(x), axes=(3, 0, 1, 2)))
            train_x_final['affinitys'].append(affin[i])
    return train_x_final


def extract_features(configs, dataset_name, idx, coords, features, columns, std, rotation):
    x = []
    coords_idx = rotate(coords[dataset_name][idx], rotation)
    features_idx = features[dataset_name][idx]

    x.append(make_grid(coords_idx, features_idx,
                       grid_resolution=configs.grid_spacing,
                       max_dist=configs.max_dist))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std
    x = np.transpose(np.squeeze(x), axes=(3, 0, 1, 2))
    return x


def get_batchs(configs, dataset_name, indices, coords, features, columns, std, affin, rotations=0):
    train_x_final = {'coords_features': [], 'affinitys': []}
    if isinstance(rotations, int):
        for i, idx in enumerate(indices):
            x = extract_features(configs, dataset_name, idx, coords, features, columns, std, rotations)
            train_x_final['coords_features'].append(x)
            train_x_final['affinitys'].append(affin[i])
    else:
        for rotation in rotations:
            for i, idx in enumerate(indices):
                x = extract_features(configs, dataset_name, idx, coords, features, columns, std, rotation)
                train_x_final['coords_features'].append(x)
                train_x_final['affinitys'].append(affin[i])
    return train_x_final


def preprocess_dataset(configs, paths, batch_rotation, batch_no_rotation, v_batch_rotation):
    """dataset preprocess"""
    datasets_stage = ['validation', 'training', 'test']
    ids = {}
    affinity = {}
    coords = {}
    features = {}
    featurizer = Featurizer()
    print("atomic properties: ", featurizer.FEATURE_NAMES)
    columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

    for dictionary in [ids, affinity, coords, features]:
        for datasets_name in datasets_stage:
            dictionary[datasets_name] = []

    paths = os.path.abspath(paths)
    for dataset_name in datasets_stage:
        dataset_path = os.path.join(paths, dataset_name + '_set.hdf')
        with h5py.File(dataset_path, 'r') as f:
            for pdb_id in f:
                dataset = f[pdb_id]
                coords[dataset_name].append(dataset[:, :3])
                features[dataset_name].append(dataset[:, 3:])
                affinity[dataset_name].append(dataset.attrs['affinity'])
                ids[dataset_name].append(pdb_id)

        ids[dataset_name] = np.array(ids[dataset_name])
        affinity[dataset_name] = np.reshape(affinity[dataset_name], (-1, 1))

    charges = []
    for feature_data in features['training']:
        charges.append(feature_data[..., columns['partialcharge']])
    charges = np.concatenate([c.flatten() for c in charges])
    charges_mean = charges.mean()
    charges_std = charges.std()
    print("charges mean=%s, std=%s" % (charges_mean, charges_std))
    print("Using charges std as scaling factor")

    # Best error we can get without any training (MSE from training set mean):
    t_baseline = ((affinity['training'] - affinity['training'].mean()) ** 2.0).mean()
    v_baseline = ((affinity['validation'] - affinity['training'].mean()) ** 2.0).mean()
    print('baseline mse: training=%s, validation=%s' % (t_baseline, v_baseline))
    ds_sizes = {dataset: len(affinity[dataset]) for dataset in datasets_stage}

    # val set
    val_y = affinity['validation']
    no_batch_size = ds_sizes['validation']
    ds_sizes_range = list(range(no_batch_size))
    val_coords_features = get_batchs(configs, 'validation', ds_sizes_range,
                                     coords, features, columns, charges_std, val_y, v_batch_rotation)

    # train set with rotation
    train_y = affinity['training']
    y_train_size = ds_sizes['training']
    train_sizes_range = list(range(y_train_size))
    train_coords_features = get_batchs(configs, 'training', train_sizes_range, coords,
                                       features, columns, charges_std, train_y, batch_rotation)
    # train set without rotation
    train_no_rotationcoords_features = get_batchs(configs, 'training', train_sizes_range, coords,
                                                  features, columns, charges_std, train_y, batch_no_rotation)

    return train_coords_features, y_train_size, train_no_rotationcoords_features, val_coords_features, no_batch_size



class DatasetIter:
    """dataset iterator"""
    def __init__(self, coor_features, affinity):
        self.coor_features = coor_features
        self.affinity = affinity

    def __getitem__(self, index):
        return self.coor_features[index], self.affinity[index]

    def __len__(self):
        return len(self.coor_features)


def minddataset_loader(configs, mindfile, no_batch_size):
    """rotation and without rotation dataset loader"""
    rank_size, rank_id = _get_rank_info()
    no_rot_weight = configs.batch_size / no_batch_size
    train_loader = ds.MindDataset(mindfile, columns_list=["coords_features", "affinitys"],
                                  num_parallel_workers=8, num_shards=rank_size, shard_id=rank_id)
    type_cast_op = C.TypeCast(mstype.float32)
    train_loader = train_loader.map(input_columns='coords_features', operations=type_cast_op)
    train_loader = train_loader.map(input_columns='affinitys', operations=type_cast_op)
    train_loader = train_loader.batch(batch_size=configs.batch_size, drop_remainder=True)
    return train_loader, no_rot_weight


def minddataset_loader_val(configs, mindfile, no_batch_size):
    """validation dataset loader"""
    no_rot_weight = configs.batch_size / no_batch_size
    train_loader = ds.MindDataset(mindfile, columns_list=["coords_features", "affinitys"],
                                  num_parallel_workers=8)
    type_cast_op = C.TypeCast(mstype.float32)
    train_loader = train_loader.map(input_columns='coords_features', operations=type_cast_op)
    train_loader = train_loader.map(input_columns='affinitys', operations=type_cast_op)
    train_loader = train_loader.batch(batch_size=configs.batch_size)
    return train_loader, no_rot_weight


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
