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
"""Anchor generator builder"""
from src.core.anchor_generator import AnchorGeneratorStride


def build(anchor_config, a_type):
    """build anchor generator"""
    ag_type = a_type

    if ag_type == 'anchor_generator_stride':
        ag = AnchorGeneratorStride(
            sizes=list(anchor_config['sizes']),
            anchor_strides=list(anchor_config['strides']),
            anchor_offsets=list(anchor_config['offsets']),
            rotations=list(anchor_config['rotations']),
            match_threshold=anchor_config['matched_threshold'],
            unmatch_threshold=anchor_config['unmatched_threshold'],
            class_id=ag_type)
        return ag
    raise ValueError(" unknown anchor generator type")
