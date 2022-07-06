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
"""TRN implementation"""

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer

from src.utils import get_frames_combinations


class KaimingUniform(initializer.Initializer):
    """Custom Kaiming initializer"""
    def __init__(self, in_channels):
        super().__init__()
        self._in_channels = in_channels
        self._boundary = 1 / np.sqrt(self._in_channels)

    def _initialize(self, arr):
        data = np.random.uniform(-self._boundary, self._boundary, arr.shape)
        arr[:] = data[:]


class CustomDense(nn.Dense):

    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels,
            out_channels,
            weight_init=KaimingUniform(in_channels),
            bias_init=KaimingUniform(in_channels),
        )


class RelationModuleMultiScale(nn.Cell):
    """Multi-scale relation module"""

    def __init__(self, img_feature_dim, num_frames, num_class, num_bottleneck=256, subsample_num=3):
        super().__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = num_bottleneck
        self.subsample_num = subsample_num  # how many relations selected to sum up
        self.scales = [i for i in range(num_frames, 1, -1)]  # generate the multiple frame relations

        self.frames_combinations_indices = [
            get_frames_combinations(num_frames, scale)
            for scale in self.scales
        ]

        # Define how many different combinations of frames samples to select for each scale.
        self.subsample_scales = [
            min(self.subsample_num, len(frames_combinations))
            for frames_combinations in self.frames_combinations_indices
        ]

        self.fc_fusion_scales = nn.CellList([
            self.fc_fusion(scale)
            for scale in self.scales
        ])

        self.num_scales = len(self.scales)

        self.available_indices_combinations_t = [
            Tensor(np.array(combs), mstype.int32)
            for combs in self.frames_combinations_indices
        ]

    def fc_fusion(self, num_frames):
        """Create a fully connected block"""
        classifier = nn.SequentialCell(
            nn.ReLU(),
            CustomDense(num_frames * self.img_feature_dim, self.num_bottleneck),
            nn.ReLU(),
            CustomDense(self.num_bottleneck, self.num_class),
        )
        return classifier

    def construct(self, x, combinations):
        """Feed forward"""
        bs = x.shape[0]
        act_all = self.fc_fusion_scales[0](x.view(bs, -1))
        input_flat = x.reshape(-1, self.img_feature_dim)

        for scale_id in range(1, self.num_scales):
            # For each scale select the frames permutation groups and then extract frames
            scale = self.scales[scale_id]

            # Frame group indices for the current TRN scale
            frame_group_combinations = combinations[:, scale_id]  # [bs, subsample_num]
            # (shape if combinations is [bs, num_scales, subsample_num])

            # Get the tensor with all available frames combinations for the current TRN scale
            available_indices_combinations = self.available_indices_combinations_t[scale_id]

            for c_index in range(self.subsample_scales[scale_id]):
                # The number of actual combinations used is limited
                # (optimization is described in the original paper).
                # Each scale has its own number of combinations.
                groups_indices = frame_group_combinations[:, c_index]  # [bs]
                frames_indices = available_indices_combinations[groups_indices]  # [bs, scale]

                frames_indices_bias = _create_bs_index(bs, scale, self.num_frames)  # [bs * scale]
                frames_indices_flat = frames_indices.reshape(-1) + frames_indices_bias  # [bs * scale]

                sub_frames_features = input_flat[frames_indices_flat].reshape(bs, -1)  # [bs, scale * img_feature_dim]
                act_relation = self.fc_fusion_scales[scale_id](sub_frames_features)  # [bs, num_class]
                act_all += act_relation

        return act_all


@ops.constexpr
def _create_bs_index(batch_size, scale, num_frames):
    index_bias = np.arange(batch_size).repeat(scale) * num_frames
    return Tensor(index_bias, mstype.int32)
