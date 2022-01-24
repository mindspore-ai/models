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
"""Box coder builder"""
from src.core.box_coders import BevBoxCoder
from src.core.box_coders import GroundBox3dCoder


def build(box_coder_config):
    """build box coder"""
    box_coder_type = box_coder_config['type']
    if box_coder_type == 'ground_box3d_coder':
        return GroundBox3dCoder(box_coder_config['linear_dim'], box_coder_config['encode_angle_vector'])
    if box_coder_type == 'bev_box_coder':
        return BevBoxCoder(
            box_coder_config['linear_dim'],
            box_coder_config['encode_angle_vector'],
            box_coder_config['z_fixed'],
            box_coder_config['h_fixed']
        )
    raise ValueError("unknown box_coder type")
