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
""" mvd.py """

import mindspore.ops as P
from mindspore.common.initializer import initializer as init
from mindspore.common.initializer import Normal, HeNormal, Zero
from mindspore import nn

from src.models.resnet import resnet50
from src.models.vib import VIB


def to_edge(input_):
    """
    rgb to grey
    """
    red = input_[:, 0, :, :]
    green = input_[:, 1, :, :]
    blue = input_[:, 2, :, :]
    grey = 0.2989 * red + 0.5870 * green + 0.1440 * blue
    grey = grey.view((grey.shape[0], 1, grey.shape[1], grey.shape[2]))
    return grey  # N x 1 x h x w


def weights_init_kaiming(module):
    """
    weight init
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.set_data(init(HeNormal(negative_slope=0, mode='fan_in'),
                                    module.weight.shape, module.weight.dtype))
    elif classname.find('Linear') != -1:
        module.weight.set_data(init(HeNormal(mode='fan_out'),
                                    module.weight.shape, module.weight.dtype))
        module.bias.set_data(init(Zero(), module.bias.shape, module.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        module.gamma.set_data(
            init(Normal(mean=0.1, sigma=0.01), module.gamma.shape, module.gamma.dtype))
        module.beta.set_data(init(Zero(), module.beta.shape, module.beta.dtype))


def weights_init_classifier(module):
    """
    weight init
    """
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        module.gamma.set_data(
            init(Normal(sigma=0.001), module.gamma.shape, module.gamma.dtype))
        if module.bias:
            module.bias.set_data(
                init(Zero(), module.bias.shape, module.bias.dtype))


class Normalize(nn.Cell):
    """
    Normalize
    """

    def __init__(self, power=2):
        super().__init__()
        self.power = power
        self.pow = P.Pow()
        self.sum = P.ReduceSum(keep_dims=True)
        self.div = P.Div()

    def construct(self, x):
        norm = self.pow(x, self.power)
        norm = self.sum(norm, 1)
        norm = self.pow(norm, 1. / self.power)
        out = self.div(x, norm)
        return out


class VisibleBackbone(nn.Cell):
    """
    Visible branch
    """

    def __init__(self, num_class=395, pretrain=""):
        super().__init__()

        self.visible = resnet50(num_class=num_class, pretrain=pretrain)

    def construct(self, x):
        x = self.visible(x)

        return x


class ThermalBackbone(nn.Cell):
    """
    Thermal branch
    """

    def __init__(self, num_class=395, pretrain=""):
        super(ThermalBackbone, self).__init__()

        self.thermal = resnet50(num_class=num_class, pretrain=pretrain)

    def construct(self, x):
        x = self.thermal(x)

        return x


class SharedBackbone(nn.Cell):
    """
    Shared branch
    """

    def __init__(self, num_class=395, pretrain=""):
        super(SharedBackbone, self).__init__()

        self.base = resnet50(num_class=num_class, pretrain=pretrain)

    def construct(self, x):
        x = self.base(x)

        return x


class MVD(nn.Cell):
    """
    Main Model
    """

    def __init__(self, num_class=395, drop=0.2, z_dim=512, pretrain=""):
        super().__init__()

        self.rgb_backbone = VisibleBackbone(num_class=num_class, pretrain=pretrain)
        self.ir_backbone = ThermalBackbone(num_class=num_class, pretrain=pretrain)
        self.shared_backbone = SharedBackbone(num_class=num_class, pretrain=pretrain)

        pool_dim = 2048
        self.rgb_bottleneck = VIB(in_ch=pool_dim, z_dim=z_dim, num_class=num_class)
        self.ir_bottleneck = VIB(in_ch=pool_dim, z_dim=z_dim, num_class=num_class)
        self.shared_bottleneck = VIB(in_ch=pool_dim, z_dim=z_dim, num_class=num_class)

        self.dropout = drop

        self.l2norm = Normalize(2)

        self.avgpool = P.ReduceMean(keep_dims=True)
        self.cat = P.Concat()
        self.cat_dim1 = P.Concat(axis=1)

    def construct(self, inputs):
        """
        Construct MVD
        """

        # the output of backbone is (feature, logits)
        v_observation = self.rgb_backbone(inputs)

        # inferred branch
        x_grey = to_edge(inputs)
        i_ms_input = self.cat_dim1([x_grey, x_grey, x_grey])

        i_observation = self.ir_backbone(i_ms_input)

        # modal shared branch
        v_ms_observation = self.shared_backbone(inputs)

        i_ms_observation = self.shared_backbone(i_ms_input)

        if self.training:
            v_representation = self.rgb_bottleneck(v_observation[0])
            i_representation = self.ir_bottleneck(i_observation[0])
            v_ms_representation = self.shared_bottleneck(v_ms_observation[0])
            i_ms_representation = self.shared_bottleneck(i_ms_observation[0])

            return v_observation, v_representation, v_ms_observation, v_ms_representation, \
                   i_observation, i_representation, i_ms_observation, i_ms_representation

        v_observation = self.l2norm(v_observation)
        v_ms_observation = self.l2norm(v_ms_observation)

        i_observation = self.l2norm(i_observation)
        i_ms_observation = self.l2norm(i_ms_observation)

        return v_observation, v_ms_observation, i_observation, i_ms_observation
