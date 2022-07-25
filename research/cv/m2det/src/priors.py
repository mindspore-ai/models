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

import math
from itertools import product

import mindspore

_NUM_REGLAYER = 6


def reglayer_scale(size, num_layer, size_the):
    reg_layer_size = []
    for i in range(num_layer + 1):
        size = math.ceil(size / 2.)
        if i >= 2:
            reg_layer_size += [size]
            if i == num_layer and size_the != 0:
                reg_layer_size += [size - size_the]
    return reg_layer_size


def get_scales(size, size_pattern):
    size_list = []
    for x in size_pattern:
        size_list += [round(x * size, 2)]
    return size_list


def prepare_aspect_ratio(num):
    as_ra = []
    for _ in range(num):
        as_ra += [[2, 3]]
    return as_ra


def mk_anchors(size, multiscale_size, size_pattern, step_pattern):
    cfg = {
        'feature_maps': reglayer_scale(size, _NUM_REGLAYER, 2),
        'min_dim': size,
        'steps': step_pattern,
        'min_sizes': get_scales(multiscale_size, size_pattern[:-1]),
        'max_sizes': get_scales(multiscale_size, size_pattern[1:]),
        'aspect_ratios': prepare_aspect_ratio(_NUM_REGLAYER),
        'variance': [0.1, 0.2],
        'clip': True,
    }
    return cfg


def anchors(cfg):
    input_size = cfg.model['input_size']
    size_pattern = cfg.model['anchor_config']['size_pattern']
    step_pattern = cfg.model['anchor_config']['step_pattern']
    return mk_anchors(input_size, input_size, size_pattern, step_pattern)


class PriorBox:
    """Computing prior boxes coordinates"""

    def __init__(self, config):
        cfg = anchors(config)
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            f_k = self.image_size / self.steps[k]
            s_k = self.min_sizes[k] / self.image_size
            s_k_prime = math.sqrt(s_k * (self.max_sizes[k] / self.image_size))

            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                mean += [cx, cy, s_k, s_k]
                mean += [cx, cy, s_k_prime, s_k_prime]  # aspect_ratio: 1

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]

        output = mindspore.Tensor(mean, dtype=mindspore.float32).view(-1, 4)
        if self.clip:
            output = mindspore.ops.clip_by_value(output, clip_value_max=1, clip_value_min=0)
        return output
