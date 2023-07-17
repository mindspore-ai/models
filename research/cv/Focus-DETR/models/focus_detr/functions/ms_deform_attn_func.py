# Copyright 2023 Huawei Technologies Co., Ltd
#
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
#
from __future__ import absolute_import, division, print_function

import mindspore.numpy as ms_np
import mindspore.ops as ops


def ms_deform_attn(value, value_spatial_shapes, sampling_locations, attention_weights):

    N_, _, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    split_points = [H_ * W_ for H_, W_ in value_spatial_shapes]
    split_points = split_points[:-1]
    split_points_size = len(split_points)
    for i in range(1, split_points_size):
        split_points[i] = split_points[i - 1] + split_points[i]
    split_points = [int(one.asnumpy()) for one in split_points]
    value_list = ms_np.split(value, split_points, axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        n1, n2, n3, n4 = value_list[lid_].shape
        value_l_ = value_list[lid_].reshape(n1, n2, n3 * n4)
        value_l_ = ops.transpose(value_l_, (0, 2, 1))
        value_l_ = value_l_.reshape(N_ * M_, D_, int(H_.asnumpy()), int(W_.asnumpy()))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]
        sampling_grid_l_ = ops.transpose(sampling_grid_l_, (0, 2, 1, 3, 4))
        n1, n2, n3, n4, n5 = sampling_grid_l_.shape
        sampling_grid_l_ = sampling_grid_l_.reshape(n1 * n2, n3, n4, n5)
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = ops.transpose(attention_weights, (0, 2, 1, 3, 4)).reshape(N_ * M_, 1, Lq_, L_ * P_)
    sampling_value_list = ms_np.stack(sampling_value_list, axis=-2)
    n1, n2, n3, n4, n5 = sampling_value_list.shape
    sampling_value_list = sampling_value_list.reshape(n1, n2, n3, n5 * n4)
    output = (sampling_value_list * attention_weights).sum(-1).view((N_, M_ * D_, Lq_))
    output = ops.transpose(output, (0, 2, 1))
    return output
