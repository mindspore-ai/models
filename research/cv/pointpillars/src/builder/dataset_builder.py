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
"""Dataset builder"""
from functools import partial

from src.builder import dbsampler_builder
from src.data.dataset import KittiDataset
from src.data.preprocess import prep_pointcloud


def get_dataset_keys(training, generate_bev):
    """get dataset keys"""
    keys = [
        'voxels', 'num_points', 'coordinates', 'num_voxels',
        'rect', 'Trv2c', 'P2', 'anchors', 'anchors_mask'
    ]
    if generate_bev:
        keys.append('bev_map')
    if training:
        keys.extend(['labels', 'reg_targets', 'reg_weights'])
    keys.extend(['image_idx', 'image_shape'])

    return keys


def build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          target_assigner=None):
    """Build a dataset"""
    generate_bev = model_config['use_bev']
    without_reflectivity = model_config['without_reflectivity']
    num_point_features = model_config['num_point_features']
    out_size_factor = model_config['rpn']['layer_strides'][0] // model_config['rpn']['upsample_strides'][0]

    cfg = input_reader_config
    db_sampler_cfg = input_reader_config.get('database_sampler')
    db_sampler = None
    if db_sampler_cfg:
        db_sampler = dbsampler_builder.build(db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    data_keys = get_dataset_keys(training, generate_bev)

    prep_func = partial(
        prep_pointcloud,
        root_path=cfg['kitti_root_path'],
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        db_sampler=db_sampler,
        max_voxels=cfg['max_number_of_voxels'],
        class_names=list(cfg['class_names']),
        remove_outside_points=False,
        training=training,
        shuffle_points=cfg['shuffle_points'],
        remove_unknown=cfg.get('remove_unknown_examples', False),
        gt_rotation_noise=list(cfg.get('groundtruth_rotation_uniform_noise', [])),
        gt_loc_noise_std=list(cfg.get('groundtruth_localization_noise_std', [])),
        global_rotation_noise=list(cfg.get('global_rotation_uniform_noise', [])),
        global_scaling_noise=list(cfg.get('global_scaling_uniform_noise', [])),
        global_loc_noise_std=(0.2, 0.2, 0.2),
        global_random_rot_range=list(cfg.get('global_random_rotation_range_per_object', [])),
        generate_bev=generate_bev,
        without_reflectivity=without_reflectivity,
        num_point_features=num_point_features,
        anchor_area_threshold=cfg['anchor_area_threshold'],
        remove_points_after_sample=cfg.get('remove_points_after_sample', False),
        remove_environment=cfg['remove_environment'],
        out_size_factor=out_size_factor
    )
    dataset = KittiDataset(
        info_path=cfg['kitti_info_path'],
        root_path=cfg['kitti_root_path'],
        num_point_features=num_point_features,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func,
        data_keys=data_keys
    )

    return dataset
