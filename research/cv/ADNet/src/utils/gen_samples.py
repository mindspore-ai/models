# Copyright 2021 Huawei Technologies Co., Ltd
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
# Generate sample bounding boxes.
# matlab code:
# https://github.com/hellbell/ADNet/blob/master/utils/gen_samples.m
import numpy as np

from src.utils.my_math import normal_round


def gen_samples(dst_type, bb, n, opts, trans_f, scale_f):
    # type => sampling method
    #    'gaussian'          generate samples from a Gaussian distribution centered at bb
    #                        -> positive samples, target candidates
    #    'uniform'           generate samples from a uniform distribution around bb (same aspect ratio)
    #                        -> negative samples
    #    'uniform_aspect'    generate samples from a uniform distribution around bb with varying aspect ratios
    #                        -> training samples for bbox regression
    #    'whole'             generate samples from the whole image
    #                        -> negative samples at the initial frame
    assert dst_type in ['gaussian', 'uniform', 'uniform_aspect', 'whole'], "type of sampling method is unavailable"

    h = opts['imgSize'][0]
    w = opts['imgSize'][1]

    # [center_x center_y width height]
    sample = [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]]
    samples = np.tile(sample, (n, 1))
    if dst_type == 'gaussian':
        samples[:, 0:2] = samples[:, 0:2] + trans_f * normal_round(np.mean(bb[2:4])) * \
                          np.maximum(-1, np.minimum(1, 0.5 * np.random.randn(n, 2)))
        samples[:, 2:4] = np.multiply(samples[:, 2:4],
                                      np.power(opts['scale_factor'], scale_f *
                                               np.maximum(-1, np.minimum(1, 0.5 * np.random.randn(n, 1)))))
    elif dst_type == 'uniform':
        samples[:, 0:2] = samples[:, 0:2] + trans_f * normal_round(np.mean(bb[2:4])) * (np.random.rand(n, 2) * 2 - 1)
        samples[:, 2:4] = np.multiply(samples[:, 2:4],
                                      np.power(opts['scale_factor'], scale_f * (np.random.rand(n, 1) * 2 - 1)))
    elif dst_type == 'uniform_aspect':
        samples[:, 0:2] = samples[:, 0:2] + trans_f * np.multiply(bb[2:4], np.random.rand(n, 2) * 2 - 1)
        samples[:, 2:4] = np.multiply(samples[:, 2:4], np.power(opts['scale_factor'], np.random.rand(n, 2) * 4 - 2))
        samples[:, 2:4] = np.multiply(samples[:, 2:4], np.power(opts['scale_factor'], scale_f * np.random.rand(n, 1)))
    else:  # elif type == 'whole'
        # TODO: I am not very sure if this is correct or not...
        range_ = np.array(normal_round([bb[2] / 2, bb[3] / 2, w - bb[2] / 2, h - bb[3] / 2])).astype(int)
        stride = np.array(normal_round([bb[2] / 5, bb[3] / 5])).astype(int)
        dx, dy, ds = np.meshgrid(range(range_[0], range_[2] + stride[0], stride[0]),
                                 range(range_[1], range_[3] + stride[1], stride[1]),
                                 range(-5, 6))
        windows = [dx, dy, bb[2] * np.power(opts['scale_factor'], ds), bb[3] * np.power(opts['scale_factor'], ds)]

        samples = []
        while len(samples) < n:
            # windows[0] = x-axis
            # windows[1] = y-axis
            # windows[2] = w
            # windows[3] = h
            # random to get x, y, w, h. Each has 34 * 51 * 11 choices (in grid). Random the grid coordinate
            random_idx = [np.random.randint(1, np.array(windows[0]).shape[0], 4),
                          np.random.randint(1, np.array(windows[0]).shape[1], 4),
                          np.random.randint(1, np.array(windows[0]).shape[2], 4)]
            sample = [windows[0][random_idx[0][0]][random_idx[1][0]][random_idx[2][0]],
                      windows[1][random_idx[0][1]][random_idx[1][1]][random_idx[2][1]],
                      windows[2][random_idx[0][2]][random_idx[1][2]][random_idx[2][2]],
                      windows[3][random_idx[0][3]][random_idx[1][3]][random_idx[2][3]]]
            samples.append(sample)
        samples = np.array(samples)

    # bound the width and height
    samples[:, 2] = np.maximum(10, np.minimum(w - 10, samples[:, 2]))
    samples[:, 3] = np.maximum(10, np.minimum(h - 10, samples[:, 3]))

    # [left top width height]
    bb_samples = np.array([samples[:, 0] - samples[:, 2] / 2,
                           samples[:, 1] - samples[:, 3] / 2,
                           samples[:, 2],
                           samples[:, 3]]).transpose()

    bb_samples[:, 0] = np.maximum(1 - bb_samples[:, 2],
                                  np.minimum(w - bb_samples[:, 2] / 2, bb_samples[:, 0]))
    bb_samples[:, 1] = np.maximum(1 - bb_samples[:, 3],
                                  np.minimum(h - bb_samples[:, 3] / 2, bb_samples[:, 1]))
    bb_samples = normal_round(bb_samples)
    bb_samples = bb_samples.astype(np.float32)

    return bb_samples
