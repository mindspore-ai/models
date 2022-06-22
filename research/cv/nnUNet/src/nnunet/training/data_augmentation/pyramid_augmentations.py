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

"""pyramid augmentation"""

from copy import deepcopy

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from skimage.morphology import label, ball
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening


class RemoveRandomConnectedComponentFromOneHotEncodingTransform(AbstractTransform):
    def __init__(self, channel_idx, key="data", p_per_sample=0.2, fill_with_other_class_p=0.25,
                 dont_do_if_covers_more_than_X_percent=0.25, p_per_label=1):
        """
        :param dont_do_if_covers_more_than_X_percent: dont_do_if_covers_more_than_X_percent=0.25 is 25!
        :param channel_idx: can be list or int
        :param key:
        """
        self.p_per_label = p_per_label
        self.dont_do_if_covers_more_than_X_percent = dont_do_if_covers_more_than_X_percent
        self.fill_with_other_class_p = fill_with_other_class_p
        self.p_per_sample = p_per_sample
        self.key = key
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in self.channel_idx:
                    if np.random.uniform() < self.p_per_label:
                        workon = np.copy(data[b, c])
                        num_voxels = np.prod(workon.shape, dtype=np.uint64)
                        lab, num_comp = label(workon, return_num=True)
                        self.part_of_remove_random_connected_componet(b, c, data, lab, num_comp, num_voxels)
        data_dict[self.key] = data
        return data_dict

    def part_of_remove_random_connected_componet(self, b, c, data, lab, num_comp, num_voxels):
        """This function is part of RemoveRandomConnectedComponentFromOneHotEncodingTransform """
        if num_comp > 0:
            component_ids = []
            component_sizes = []
            for i in range(1, num_comp + 1):
                component_ids.append(i)
                component_sizes.append(np.sum(lab == i))
            component_ids = [i for i, j in zip(component_ids, component_sizes) if
                             j < num_voxels * self.dont_do_if_covers_more_than_X_percent]

            if component_ids:
                random_component = np.random.choice(component_ids)
                data[b, c][lab == random_component] = 0
                if np.random.uniform() < self.fill_with_other_class_p:
                    other_ch = [i for i in self.channel_idx if i != c]
                    if other_ch:
                        other_class = np.random.choice(other_ch)
                        data[b, other_class][lab == random_component] = 1


class MoveSegAsOneHotToData(AbstractTransform):
    """move segmentaiton as onehot to data"""

    def __init__(self, channel_id, all_seg_labels, key_origin="seg", key_target="data", remove_from_origin=True):
        self.remove_from_origin = remove_from_origin
        self.all_seg_labels = all_seg_labels
        self.key_target = key_target
        self.key_origin = key_origin
        self.channel_id = channel_id

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)
        seg = origin[:, self.channel_id:self.channel_id + 1]
        seg_onehot = np.zeros((seg.shape[0], len(self.all_seg_labels), *seg.shape[2:]), dtype=seg.dtype)
        for i, l in enumerate(self.all_seg_labels):
            seg_onehot[:, i][seg[:, 0] == l] = 1
        target = np.concatenate((target, seg_onehot), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            remaining_channels = [i for i in range(origin.shape[1]) if i != self.channel_id]
            origin = origin[:, remaining_channels]
            data_dict[self.key_origin] = origin
        return data_dict


class ApplyRandomBinaryOperatorTransform(AbstractTransform):
    """Apply RandomBinary Operator Transform module"""

    def __init__(self, channel_idx, p_per_sample=0.3, any_of_these=(binary_dilation, binary_erosion, binary_closing,
                                                                    binary_opening),
                 key="data", strel_size=(1, 10), p_per_label=1):
        self.p_per_label = p_per_label
        self.strel_size = strel_size
        self.key = key
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample

        assert not isinstance(channel_idx, tuple), "bäh"

        if not isinstance(channel_idx, list):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                ch = deepcopy(self.channel_idx)
                np.random.shuffle(ch)
                for c in ch:
                    self.part_of_apply_random_binary_operator_transform(b, c, ch, data)
        data_dict[self.key] = data
        return data_dict

    def part_of_apply_random_binary_operator_transform(self, b, c, ch, data):
        """This function is part of apply random binary transform"""
        if np.random.uniform() < self.p_per_label:
            operation = np.random.choice(self.any_of_these)
            selem = ball(np.random.uniform(*self.strel_size))
            workon = np.copy(data[b, c]).astype(int)
            res = operation(workon, selem).astype(workon.dtype)
            data[b, c] = res

            # if class was added, we need to remove it in ALL other channels to keep one hot encoding
            # properties
            # we modify data
            other_ch = [i for i in ch if i != c]
            if other_ch:
                was_added_mask = (res - workon) > 0
                for oc in other_ch:
                    data[b, oc][was_added_mask] = 0
                # if class was removed, leave it at background


class ApplyRandomBinaryOperatorTransform2(AbstractTransform):
    """Apply Random Binary OperatorTransform2"""

    def __init__(self, channel_idx, p_per_sample=0.3, p_per_label=0.3, any_of_these=(binary_dilation, binary_closing),
                 key="data", strel_size=(1, 10)):

        self.strel_size = strel_size
        self.key = key
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample
        self.p_per_label = p_per_label

        assert not isinstance(channel_idx, tuple), "bäh"

        if not isinstance(channel_idx, list):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                ch = deepcopy(self.channel_idx)
                np.random.shuffle(ch)
                for c in ch:
                    self.part_of_applyrandombinary(b, c, ch, data)
        data_dict[self.key] = data
        return data_dict

    def part_of_applyrandombinary(self, b, c, ch, data):
        """This function is part of apply random binary"""
        if np.random.uniform() < self.p_per_label:
            operation = np.random.choice(self.any_of_these)
            selem = ball(np.random.uniform(*self.strel_size))
            workon = np.copy(data[b, c]).astype(int)
            res = operation(workon, selem).astype(workon.dtype)
            data[b, c] = res

            # if class was added, we need to remove it in ALL other channels to keep one hot encoding
            # properties
            # we modify data
            other_ch = [i for i in ch if i != c]
            if other_ch:
                was_added_mask = (res - workon) > 0
                for oc in other_ch:
                    data[b, oc][was_added_mask] = 0
                # if class was removed, leave it at background
