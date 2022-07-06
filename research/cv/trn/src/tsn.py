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
"""TSN implementation"""
from mindspore import nn


class TSN(nn.Cell):
    """Temporal segmentation network class

    Args:
        base_network (nn.Cell): model backbone
        consensus_network (nn.Cell): model head
    """
    def __init__(self, base_network, consensus_network):
        super().__init__()
        self.base_model = base_network
        self.consensus = consensus_network
        self.sample_len = 3  # RGB
        self.num_segments = consensus_network.num_frames

    def construct(self, x, combinations):
        """Feed forward"""
        base_out = self.base_model(x.view((-1, self.sample_len) + x.shape[2:]))
        base_out = base_out.view((-1, self.num_segments) + base_out.shape[1:])
        combinations_view = combinations.view((-1,) + combinations.shape[-2:])
        output = self.consensus(base_out, combinations_view)
        return output
