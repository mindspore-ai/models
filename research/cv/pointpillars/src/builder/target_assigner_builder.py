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
"""target assigner builder"""
from src.builder import anchor_generator_builder
from src.builder import similarity_calculator_builder
from src.core.target_assigner import TargetAssigner


def build(target_assigner_config, box_coder):
    """build target assigner"""
    anchor_cfg = target_assigner_config['anchor_generators']
    anchor_generators = []
    for cl in anchor_cfg:
        for a_type, a_cfg in anchor_cfg[cl].items():
            anchor_generator = anchor_generator_builder.build(a_cfg, a_type)
            anchor_generators.append(anchor_generator)
    similarity_calc = similarity_calculator_builder.build(
        target_assigner_config['region_similarity_calculator']
    )
    positive_fraction = target_assigner_config['sample_positive_fraction']
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config['sample_size'])
    return target_assigner
