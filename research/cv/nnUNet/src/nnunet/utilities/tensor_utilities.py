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

"""tensor utilities module"""

import numpy as np
import mindspore


def sum_tensor(inp, axes, keepdims=False):
    """sum tensor according axis 2,3,4"""
    inp = inp.sum(axis=4)
    inp = inp.sum(axis=3)
    inp = inp.sum(axis=2)

    return inp


def sum_tensor_axis_321(inp, axes, keepdims=False):
    """sum tensor according axis 2,3,4"""
    inp = inp.sum(axis=3)
    inp = inp.sum(axis=2)
    inp = inp.sum(axis=1)

    return inp


def sum_tensor_axis_21(inp, axes, keepdims=False):
    """sum tensor according axis 2,1"""
    inp = inp.sum(axis=2)
    inp = inp.sum(axis=1)

    return inp


def sum_tensor_axis_023(inp, axes, keepdims=False):
    """sum tensor according axis 0,2,3"""
    inp = inp.sum(axis=3)
    inp = inp.sum(axis=2)
    inp = inp.sum(axis=0)

    return inp


def mean_tensor(inp, axes, keepdim=False):
    """mean tensor"""
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.mean(int(ax))
    return inp


def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = mindspore.numpy.arange(x.size(dim) - 1, -1, -1,
                                          dtype=mindspore.long, device=x.device)
    return x[tuple(indices)]
