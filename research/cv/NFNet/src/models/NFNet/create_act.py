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
"""
Activation Factory Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Union, Callable, Type

from mindspore import nn


class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


_ACT_LAYER_DEFAULT = dict(
    relu=nn.ReLU(),
    relu6=nn.ReLU6(),
    leaky_relu=nn.LeakyReLU(),
    elu=nn.ELU(),
    gelu=nn.GELU(),
    sigmoid=nn.Sigmoid(),
    tanh=nn.Tanh(),
    silu=SiLU(),
)


def get_act_fn(name: Union[Callable, str] = 'relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if isinstance(name, Callable):
        return name

    return _ACT_LAYER_DEFAULT[name]


def get_act_layer(name: Union[Type[nn.Cell], str] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if isinstance(name, nn.Cell):
        return name
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name: Union[nn.Cell, str]):
    if not isinstance(name, str):
        return name
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    return act_layer
