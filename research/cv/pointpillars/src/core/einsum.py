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
"""einsum"""
from collections import OrderedDict

import numpy as np
from mindspore import ops

VALID_LABELS = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def parse_format(f):
    """parse format"""
    if '->' not in f:
        raise ValueError('incorrect format received')

    f_inputs, f_output = f.split('->')

    if not f_inputs:
        raise ValueError

    f_inputs = [list(f) for f in f_inputs.split(',')]
    f_output = list(f_output)

    if len(set(f_output)) != len(f_output):
        raise ValueError(f'duplicate label in f_output: {f_output}')

    for f_input in f_inputs:
        if set(f_input) > VALID_LABELS:
            raise ValueError
        if len(set(f_input)) < len(f_input):
            raise ValueError(f"duplicate label {f_input}")

    return f_inputs, f_output


def validate_args(f_inputs, tensors):
    """validate args"""
    assert len(tensors) == len(f_inputs)

    dimensions = OrderedDict()
    for t in range(len(tensors)):
        fmt = f_inputs[t]
        assert tensors[t].ndim == len(fmt)

        for i in range(len(fmt)):
            if fmt[i] in dimensions:
                assert dimensions[fmt[i]] == tensors[t].shape[i]
            else:
                dimensions[fmt[i]] = tensors[t].shape[i]

    return dimensions


def transpose(tensor, permutation):
    """transpose"""
    if isinstance(tensor, np.ndarray):
        return np.transpose(tensor, permutation)
    return tensor.transpose(permutation)


def outer_product(f_inputs, dimensions, tensors):
    """outer product"""
    tensors = list(tensors)
    assert len(f_inputs) == len(tensors)
    f_output = list(dimensions.keys())

    normalized = []

    while tensors:
        tensor = tensors.pop()
        labels = f_inputs.pop()

        if labels == f_output:
            normalized.append(tensor)
            continue

        source = dict(zip(labels, range(len(labels))))
        permutation = [source[l] for l in f_output if l in labels]
        labels = [labels[axis] for axis in permutation]
        tensor = ops.Transpose()(tensor, tuple(permutation))

        i = 0
        while i < len(dimensions):
            if i == len(labels) or labels[i] != f_output[i]:
                tensor = ops.ExpandDims()(tensor, i)
                labels.insert(i, f_output[i])
            else:
                i += 1

        normalized.append(tensor)

    op = normalized.pop()
    while normalized:
        tensor = normalized.pop()
        op = op * tensor

    return op


def contract(op, dimensions, f_output):
    """contract"""
    if not f_output:
        return op.sum()

    f_input = list(dimensions.keys())
    axis = 0
    while op.ndim > len(f_output):
        assert len(f_input) == op.ndim
        if f_input[axis] not in f_output:
            op = op.sum(axis)
            del f_input[axis]
        else:
            axis += 1

    if f_input == f_output:
        return op
    source = dict(zip(f_input, range(len(f_input))))
    permutation = [source[l] for l in f_output]
    return ops.Transpose()(op, tuple(permutation))


def einsum(f, *tensors):
    """einsum"""
    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(f_inputs, tensors)

    op = outer_product(f_inputs, dimensions, tensors)
    assert op.shape == tuple(dimensions.values())
    contraction = contract(op, dimensions, f_output)
    return contraction
