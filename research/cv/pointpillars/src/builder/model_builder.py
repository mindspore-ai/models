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
"""Model builder"""
from src.pointpillars import PointPillarsNet


def build(model_cfg, voxel_generator, target_assigner):
    """build model"""
    vfe_num_filters = model_cfg['voxel_feature_extractor']['num_filters']
    grid_size = voxel_generator.grid_size
    output_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    voxel_size = list(map(float, voxel_generator.voxel_size))
    pc_range = list(map(float, voxel_generator.point_cloud_range))
    pointpillarsnet = PointPillarsNet(
        output_shape=output_shape,
        num_class=model_cfg['num_class'],
        num_input_features=model_cfg['num_point_features'],
        vfe_num_filters=model_cfg['voxel_feature_extractor']['num_filters'],
        with_distance=model_cfg['voxel_feature_extractor']['with_distance'],
        rpn_layer_nums=model_cfg['rpn']['layer_nums'],
        rpn_layer_strides=model_cfg['rpn']['layer_strides'],
        rpn_num_filters=model_cfg['rpn']['num_filters'],
        rpn_upsample_strides=model_cfg['rpn']['upsample_strides'],
        rpn_num_upsample_filters=model_cfg['rpn']['num_upsample_filters'],
        use_norm=model_cfg['use_norm'],
        use_direction_classifier=model_cfg['use_direction_classifier'],
        encode_background_as_zeros=model_cfg['encode_background_as_zeros'],
        num_anchor_per_loc=target_assigner.num_anchors_per_location,
        code_size=target_assigner.box_coder.code_size,
        use_bev=model_cfg['use_bev'],
        voxel_size=voxel_size,
        pc_range=pc_range
    )

    return pointpillarsnet
