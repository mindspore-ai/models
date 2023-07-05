# Copyright 2023 Huawei Technologies Co., Ltd
#
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore import numpy as mnp


class MLP(nn.Cell):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([nn.Dense(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def construct(self, x):
        """construct"""
        for i, layer in enumerate(self.layers):
            x = ops.ReLU()(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return nn.SeLU()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, learnedwh=None):
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_ = memory.shape[0]
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        H_ = int(H_.asnumpy())
        W_ = int(W_.asnumpy())
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        mask_flatten_ = mask_flatten_.astype("bool")
        not_mask = ~mask_flatten_
        valid_H = not_mask[:, :, 0, 0].sum(axis=1)
        valid_W = not_mask[:, 0, :, 0].sum(axis=1)

        start = Tensor(0, ms.float32)
        end = Tensor(H_ - 1, ms.float32)
        h = H_
        H_m = ops.linspace(start, end, h)
        start = Tensor(0, ms.float32)
        end = Tensor(W_ - 1, ms.float32)
        w = W_
        W_m = ops.linspace(start, end, w)
        grid_y, grid_x = ops.meshgrid((H_m, W_m), indexing="ij")
        expand_dims = ops.ExpandDims()
        grid = ops.concat([expand_dims(grid_x, -1), expand_dims(grid_y, -1)], -1)
        scale = ops.concat([expand_dims(valid_W, -1), expand_dims(valid_H, -1)], 1).view(N_, 1, 1, 2)
        grid = expand_dims(grid, 0)
        grid = (ops.broadcast_to(grid, (N_, -1, -1, -1)) + 0.5) / scale
        if learnedwh is not None:
            wh = ops.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = ops.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = ops.concat([grid, wh], -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_
    output_proposals = ops.concat(proposals, 1)
    output_proposals_1 = output_proposals > 0.01
    output_proposals_2 = output_proposals < 0.99
    output_proposals_valid = ops.logical_and(output_proposals_1, output_proposals_2)
    output_proposals_valid = output_proposals_valid.all(-1, keep_dims=True)
    output_proposals = ops.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = ops.masked_fill(output_proposals, expand_dims(memory_padding_mask, -1), float("inf"))
    output_proposals = ops.masked_fill(output_proposals, ~output_proposals_valid, float("inf"))
    output_memory = memory
    output_memory = ops.masked_fill(output_memory, expand_dims(memory_padding_mask, -1), float(0))
    output_memory = ops.masked_fill(output_memory, ~output_proposals_valid, float(0))

    return output_memory, output_proposals


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = mnp.arange(128, dtype=ms.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = ops.Stack(axis=3)((ops.Sin()(pos_x[:, :, 0::2]), ops.Cos()(pos_x[:, :, 1::2])))

    pos_x = pos_x.view(*pos_x.shape[:2], -1)
    pos_y = ops.Stack(axis=3)((ops.Sin()(pos_y[:, :, 0::2]), ops.Cos()(pos_y[:, :, 1::2])))
    pos_y = pos_y.view(*pos_y.shape[:2], -1)
    if pos_tensor.shape[-1] == 2:
        pos = ops.Concat(axis=2)((pos_y, pos_x))
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = ops.Stack(axis=3)((ops.Sin()(pos_w[:, :, 0::2]), ops.Cos()(pos_w[:, :, 1::2])))

        pos_w = pos_w.view(*pos_w.shape[:2], -1)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = ops.Stack(axis=3)((ops.Sin()(pos_h[:, :, 0::2]), ops.Cos()(pos_h[:, :, 1::2])))
        pos_h = pos_h.view(*pos_h.shape[:2], -1)
        pos = ops.Concat(axis=2)((pos_y, pos_x, pos_w, pos_h))
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


def inverse_sigmoid(x, eps=0.001):
    min_value = ms.Tensor(0, ms.float32)
    max_value = ms.Tensor(1, ms.float32)
    x = ops.clip_by_value(x, min_value, max_value)

    mini_value = ms.Tensor(eps, ms.float32)

    x1 = ops.clip_by_value(x, mini_value, max_value)
    x2 = ops.clip_by_value((1 - x), mini_value, max_value)

    return ops.log(x1 / x2)
