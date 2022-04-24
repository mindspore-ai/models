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
"""similarity calculator builder"""
from src.core import region_similarity


def build(similarity_config):
    """build similarity config"""
    similarity_type = similarity_config
    if similarity_type == 'rotate_iou_similarity':
        return region_similarity.RotateIouSimilarity()
    if similarity_type == 'nearest_iou_similarity':
        return region_similarity.NearestIouSimilarity()
    if similarity_type == 'distance_similarity':
        cfg = similarity_config.distance_similarity
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha)
    raise ValueError("unknown similarity type")
