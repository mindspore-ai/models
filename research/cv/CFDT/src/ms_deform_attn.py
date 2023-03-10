# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
# This file was modified from arcgis for CFDT
# https://pypi.org/project/arcgis/
# ----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import mindspore as ms
from mindspore import ops, nn
from mindspore import numpy as msnp


class grid_sample(nn.Cell):
    def __init__(self):
        super(grid_sample, self).__init__()
        self.gather = ops.GatherNd()
        self.concat = ops.Concat(1)

    def construct(self, input_tens, grid):
        B, C, H, W = input_tens.shape
        _, IH, IW, _ = grid.shape
        B_ind = ops.cast(msnp.arange(B).repeat(
            C * IH * IW), ms.int32).reshape((-1, 1))
        C_ind = ops.cast(msnp.arange(C).repeat(
            IH * IW), ms.int32).reshape((-1, 1))
        C_ind = ops.Tile()(C_ind, (B, 1))
        iy_temp = (((grid[..., 1] + 1) * H - 1) / 2).reshape((B, -1))
        ix_temp = (((grid[..., 0] + 1) * W - 1) / 2).reshape((B, -1))
        iy = ops.repeat_elements(iy_temp, rep=C, axis=0).reshape(-1, 1)
        ix = ops.repeat_elements(ix_temp, rep=C, axis=0).reshape(-1, 1)
        ix_nw = ops.floor(ix)  # , W)
        iy_nw = ops.floor(iy)  # , H)
        ix_se = ix_nw + 1  # , W)
        iy_se = iy_nw + 1  # , H)
        nw_ind = self.concat((B_ind, C_ind, ops.cast(
            iy_nw, ms.int32), ops.cast(ix_nw, ms.int32)))
        nw = self.gather(input_tens, nw_ind)
        ne_ind = self.concat((B_ind, C_ind, ops.cast(
            iy_nw, ms.int32), ops.cast(ix_se, ms.int32)))
        ne = self.gather(input_tens, ne_ind)
        sw_ind = self.concat((B_ind, C_ind, ops.cast(
            iy_se, ms.int32), ops.cast(ix_nw, ms.int32)))
        sw = self.gather(input_tens, sw_ind)
        se_ind = self.concat((B_ind, C_ind, ops.cast(
            iy_se, ms.int32), ops.cast(ix_se, ms.int32)))
        se = self.gather(input_tens, se_ind)

        nw_w = ops.absolute(((ix_se - ix) * (iy_se - iy)).reshape((-1,)))
        ne_w = ops.absolute(((ix - ix_nw) * (iy_se - iy)).reshape((-1,)))
        sw_w = ops.absolute(((ix_se - ix) * (iy - iy_nw)).reshape((-1,)))
        se_w = ops.absolute(((ix - ix_nw) * (iy - iy_nw)).reshape((-1,)))

        output = nw_w * nw + ne_w * ne + sw_w * sw + se_w * se

        return output.reshape((B, C, IH, IW))


def ms_deform_attn_core(value_inner, value_spatial_shapes, sampling_locations, attention_weights, op_type=None):
    N_, _, M_, D_ = value_inner.shape  # 2,13044,8,32
    _, Lq_, M_, _, _, _ = sampling_locations.shape  # 2,100,8,4,4,2

    indices = [int(H_ * W_) for H_, W_ in value_spatial_shapes]
    sections = [indices[0], indices[0] + indices[1],
                indices[0] + indices[1] + indices[2]]
    value_list = msnp.split(value_inner, sections, axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, batch in enumerate(value_spatial_shapes):
        H_, _ = batch
        z_, o_ = value_list[lid_].shape[0], value_list[lid_].shape[1]
        value_l_temp = ops.Reshape()(
            value_list[lid_], (z_, o_, -1)).transpose(0, 2, 1)
        value_l_ = ops.Reshape()(value_l_temp, (N_ * M_, D_, int(H_), -1))
        temp_grids = sampling_grids[:, :, :, lid_].transpose(0, 2, 1, 3, 4)
        sampling_grid_l_ = ops.Reshape()(temp_grids,
                                         (-1, temp_grids.shape[2], temp_grids.shape[3], temp_grids.shape[4]))
        sampling_value_l_ = grid_sample()(value_l_, sampling_grid_l_)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = ops.Reshape()(
        attention_weights.transpose(0, 2, 1, 3, 4), (N_ * M_, 1, Lq_, -1))
    if op_type == 'inner':
        temp_sampling_value = ops.stack(sampling_value_list, axis=-2)
        sampling_value = temp_sampling_value.reshape(temp_sampling_value.shape[0], temp_sampling_value.shape[1],
                                                     temp_sampling_value.shape[2], -1)
        output = (attention_weights *
                  sampling_value).sum(-1).view(N_, M_ * D_, Lq_)
    else:
        temp = ops.stack(sampling_value_list, axis=-2)
        output = (temp.reshape(temp.shape[0], temp.shape[1], temp.shape[2], -1)
                  * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    output = ops.Transpose()(output, (0, 2, 1))

    return output


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.kv = nn.Dense(
            in_channels=dim, out_channels=dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.norm1 = nn.LayerNorm(normalized_shape=[self.dim], epsilon=1e-05)
        self.norm2 = nn.LayerNorm(normalized_shape=[self.dim], epsilon=1e-05)

    def construct(self, x):
        B, N, C = x.shape

        x_ = x

        q = self.q(x_).reshape(B, N, self.num_heads, C //
                               self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C //
                                 self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_ = self.proj(x_)
        x_ = self.proj_drop(x_)
        x = x + x_
        return x


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Cell):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Dense(in_channels=d_model, out_channels=n_heads * n_levels * n_points * 2,
                                         weight_init='Zero', has_bias=True)
        self.attention_weights = nn.Dense(in_channels=d_model, out_channels=n_heads * n_levels * n_points,
                                          weight_init='xavier_uniform', has_bias=True)

        self.value_proj = nn.Dense(
            in_channels=d_model, out_channels=d_model, has_bias=True, bias_init='zeros')
        self.output_proj = nn.Dense(
            in_channels=d_model, out_channels=d_model, has_bias=True, bias_init='zeros')

        self.n_heads_inner = n_heads
        self.d_model_inner = d_model // 16
        self.n_points_inner = n_points * 16
        self.sampling_offsets_inner = nn.Dense(in_channels=d_model,
                                               out_channels=self.n_heads_inner * n_levels * self.n_points_inner * 2,
                                               weight_init='xavier_uniform', has_bias=True)
        self.attention_weights_inner = nn.Dense(in_channels=d_model,
                                                out_channels=self.n_heads_inner * n_levels * self.n_points_inner,
                                                weight_init='xavier_uniform', has_bias=True, bias_init='zeros')
        self.value_proj_inner = nn.Dense(in_channels=self.d_model_inner, out_channels=self.d_model_inner, has_bias=True,
                                         weight_init='zeros', bias_init='zeros')
        self.output_proj_inner = nn.Dense(in_channels=d_model // 16, out_channels=d_model, has_bias=True,
                                          weight_init='zeros', bias_init='zeros')

    def construct(self, query, reference_points, input_flatten, input_flatten_inner, input_spatial_shapes,
                  input_spatial_shapes_inner, input_level_start_index, input_level_start_index_inner,
                  input_padding_mask=None, input_padding_mask_inner=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        N, Len_in_inner, _ = input_flatten_inner.shape
        value = self.value_proj(input_flatten)
        value_inner = self.value_proj_inner(input_flatten_inner)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        if input_padding_mask_inner is not None:
            value_inner = value_inner.masked_fill(
                input_padding_mask_inner[..., None], float(0))
        value = ops.Reshape()(value, (N, Len_in, self.n_heads, self.d_model // self.n_heads,))
        value_inner = ops.Reshape()(value_inner,
                                    (N, Len_in_inner, self.n_heads_inner, self.d_model_inner // self.n_heads_inner,))
        sampling_offsets_inner = self.sampling_offsets_inner(query).view(N, Len_q, self.n_heads_inner, self.n_levels,
                                                                         self.n_points_inner, 2)
        attention_weights = self.attention_weights_inner(query).view(N, Len_q, self.n_heads_inner,
                                                                     self.n_levels * self.n_points_inner)
        attention_weights = ops.Reshape()(ops.Softmax()(attention_weights),
                                          (N, Len_q, self.n_heads_inner, self.n_levels, self.n_points_inner,))
        if reference_points.shape[-1] == 2:
            offset_normalizer_inner = ops.stack(
                [input_spatial_shapes_inner[..., 1], input_spatial_shapes_inner[..., 0]], -1)
            sampling_locations_inner = reference_points[:, :, None, :, None, :] \
                + sampling_offsets_inner / \
                offset_normalizer_inner[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations_inner = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets_inner / self.n_points_inner * reference_points[:, :, None, :,
                                                                                  None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = ms_deform_attn_core(
            value_inner, input_spatial_shapes_inner, sampling_locations_inner, attention_weights, op_type='outer')
        output = self.output_proj_inner(output)
        query = query + output
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points,
                                                             2)  # outer
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)

        attention_weights = ops.Reshape()(ops.Softmax(-1)(attention_weights),
                                          (N, Len_q, self.n_heads, self.n_levels, self.n_points,))

        if reference_points.shape[-1] == 2:
            offset_normalizer = ops.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points * \
                reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights, op_type='inner')
        output = self.output_proj(output)

        return output
