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

# This file was copied from project [kenshohara][3D-ResNets-PyTorch]

import random
import numpy as np


class LoopPadding:

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop:
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop:
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - self.size)
        end_index = min(begin_index + self.size * 2, len(frame_indices))

        out = frame_indices[begin_index:end_index:2]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop:
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size * 2 + 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size * 2, len(frame_indices))

        out = frame_indices[begin_index:end_index:2]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalMultiCrop:

    def __init__(self, size, K):
        self.size = size
        self.K = K

    def __call__(self, frame_indices):
        centers = [int(idx) for idx in np.linspace(self.size, len(frame_indices) - self.size, self.K)]
        clips = []
        for c in centers:
            begin = max(0, c - self.size)
            end = min(c + self.size, len(frame_indices))
            clip = frame_indices[begin:end:2]
            for index in clip:
                if len(clip) >= self.size:
                    break
                clip.append(index)
            clips.append(clip)
        return clips
