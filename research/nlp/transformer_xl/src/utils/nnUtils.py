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

import mindspore
from mindspore import Tensor
from mindspore.common import initializer as init


def uniform_(tensor, a=0., b=1.):
    # r"""Fills the input Tensor with values drawn from the uniform
    #     distribution :math:`\mathcal{U}(a, b)`.
    #
    #     Args:
    #         tensor: an n-dimensional `torch.Tensor`
    #         a: the lower bound of the uniform distribution
    #         b: the upper bound of the uniform distribution
    #
    #     Examples:
    #         >>> w = torch.empty(3, 5)
    #         >>> nn.init.uniform_(w)
    #     """
    tensor += Tensor(dtype=mindspore.float32, init=init.Zero(), shape=tensor.shape).fill((b - a) / 2)
    init.Uniform((b - a) / 2)(tensor.asnumpy())


def normal_(tensor, mean=0., std=1.):
    r"""Fills the input Tensor with values drawn from the normal
        distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

        Args:
            tensor: an n-dimensional `torch.Tensor`
            mean: the mean of the normal distribution
            std: the standard deviation of the normal distribution

        Examples:
            #>>> w = torch.empty(3, 5)
            #>>> nn.init.normal_(w)
    """

    init.Normal(mean=mean, sigma=std)(tensor.asnumpy())


def constant_(tensor, val):
    r"""Fills the input Tensor with the value :math:`\text{val}`.

        Args:
            tensor: an n-dimensional `torch.Tensor`
            val: the value to fill the tensor with

        Examples:
            #>>> w = torch.empty(3, 5)
            #>>> nn.init.constant_(w, 0.3)
    """
    constant_init = init.Constant(value=val)
    constant_init(tensor.asnumpy())
