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
""" Normalization Free Nets. NFNet, NF-RegNet, NF-ResNet (pre-activation) Models

Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692

Paper: `High-Performance Large-Scale Image Recognition Without Normalization`
    - https://arxiv.org/abs/2102.06171

Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Status:
* These models are a work in progress, experiments ongoing.
* Pretrained weights for two models so far, more to come.
* Model details updated to closer match official JAX code now that it's released
* NF-ResNet, NF-RegNet-B, and NFNet-F models supported

Hacked together by / copyright Ross Wightman, 2021.
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Optional

import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype

from src.models.NFNet.SEBlock import SEModule
from src.models.NFNet.classifier import ClassifierHead
from src.models.NFNet.create_act import get_act_fn, get_act_layer
from src.models.NFNet.layers import DropPath2D
from src.models.NFNet.std_conv import ScaledStdConv2dSame
from src.models.helpers import make_divisible
from src.tools.var_init import RandomNormal

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _dcfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    nfnet_f0=_dcfg(
        url='', pool_size=(6, 6), input_size=(3, 192, 192), test_input_size=(3, 256, 256)),
    nfnet_f1=_dcfg(
        url='', pool_size=(7, 7), input_size=(3, 224, 224), test_input_size=(3, 320, 320)),
    nfnet_f2=_dcfg(
        url='', pool_size=(8, 8), input_size=(3, 256, 256), test_input_size=(3, 352, 352)),
    nfnet_f3=_dcfg(
        url='', pool_size=(10, 10), input_size=(3, 320, 320), test_input_size=(3, 416, 416)),
    nfnet_f4=_dcfg(
        url='', pool_size=(12, 12), input_size=(3, 384, 384), test_input_size=(3, 512, 512)),
    nfnet_f5=_dcfg(
        url='', pool_size=(13, 13), input_size=(3, 416, 416), test_input_size=(3, 544, 544)),
    nfnet_f6=_dcfg(
        url='', pool_size=(14, 14), input_size=(3, 448, 448), test_input_size=(3, 576, 576)),
    nfnet_f7=_dcfg(
        url='', pool_size=(15, 15), input_size=(3, 480, 480), test_input_size=(3, 608, 608)),
)


@dataclass
class NfCfg:
    """build NFNet config"""
    depths: Tuple[int, int, int, int]
    channels: Tuple[int, int, int, int]
    alpha: float = 0.2
    stem_type: str = '3x3'
    stem_chs: Optional[int] = None
    group_size: Optional[int] = None
    attn_layer: Optional[str] = None
    attn_kwargs: dict = None
    attn_gain: float = 2.0  # NF correction gain to apply if attn layer is used
    width_factor: float = 1.0
    bottle_ratio: float = 0.5
    num_features: int = 0  # num out_channels for final conv, no final_conv if 0
    ch_div: int = 8  # round channels % 8 == 0 to keep tensor-core use optimal
    reg: bool = False  # enables EfficientNet-like options used in RegNet variants, expand from in_chs, se in middle
    extra_conv: bool = False  # extra 3x3 bottleneck convolution for NFNet models
    gamma_in_act: bool = False
    same_padding: bool = False
    std_conv_eps: float = 1e-5
    skipinit: bool = False  # disabled by default, non-trivial performance impact
    zero_init_fc: bool = False
    act_layer: str = 'silu'


def _nfnet_cfg(depths, channels=(256, 512, 1536, 1536), group_size=128, bottle_ratio=0.5, feat_mult=2.,
               act_layer='gelu', attn_layer='se', attn_kwargs=None):
    num_features = int(channels[-1] * feat_mult)
    attn_kwargs = attn_kwargs if attn_kwargs is not None else dict(rd_ratio=0.5)
    cfg = NfCfg(depths=depths, channels=channels, stem_type='deep_quad', stem_chs=128, group_size=group_size,
                bottle_ratio=bottle_ratio, extra_conv=True, num_features=num_features, act_layer=act_layer,
                attn_layer=attn_layer, attn_kwargs=attn_kwargs)
    return cfg


def _dm_nfnet_cfg(depths, channels=(256, 512, 1536, 1536), act_layer='gelu', skipinit=True):
    cfg = NfCfg(
        depths=depths, channels=channels, stem_type='deep_quad', stem_chs=128, group_size=128,
        bottle_ratio=0.5, extra_conv=True, gamma_in_act=True, same_padding=True, skipinit=skipinit,
        num_features=int(channels[-1] * 2.0), act_layer=act_layer, attn_layer='se', attn_kwargs=dict(rd_ratio=0.5))
    return cfg


class GammaAct(nn.Cell):
    def __init__(self, act_type='relu', gamma: float = 1.0):
        super().__init__()
        self.act_fn = get_act_fn(act_type)
        print(self.act_fn)
        self.gamma = gamma

    def construct(self, x):
        return self.act_fn(x) * self.gamma


def act_with_gamma(act_type, gamma: float = 1.):
    return GammaAct(act_type, gamma=gamma)


class DownsampleAvg(nn.Cell):
    """ AvgPool Downsampling as in 'D' ResNet variants. Support for dilation."""

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, conv_layer=ScaledStdConv2dSame):
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            self.pool = nn.AvgPool2d(2, avg_stride, pad_mode="same") if avg_stride == 1 and dilation > 1 else \
                nn.AvgPool2d(kernel_size=2, stride=avg_stride, pad_mode="valid")
        else:
            self.pool = None
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)

    def construct(self, x):
        if self.pool is None:
            return self.conv(x)
        return self.conv(self.pool(x))


class NormFreeBlock(nn.Cell):
    """Normalization-Free pre-activation block.
    """

    def __init__(self, in_chs, out_chs=None, stride=1, dilation=1, first_dilation=None,
                 alpha=0.2, beta=1.0, bottle_ratio=0.25, group_size=None, ch_div=1, reg=True, extra_conv=False,
                 skipinit=False, attn_layer=None, attn_gain=2.0, act_layer=None, conv_layer=None, drop_path_rate=0.25):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        # RegNet variants scale bottleneck from in_chs, otherwise scale from out_chs like ResNet
        mid_chs = make_divisible(in_chs * bottle_ratio if reg else out_chs * bottle_ratio, ch_div)
        group = 1 if not group_size else mid_chs // group_size
        if group_size and group_size % ch_div == 0:
            mid_chs = group_size * group  # correct mid_chs if group_size divisible by ch_div, otherwise error
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain

        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(
                in_chs, out_chs, stride=stride, dilation=dilation, conv_layer=conv_layer)
        else:
            self.downsample = None

        self.act1 = act_layer
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, group=group)
        if extra_conv:
            self.act2b = act_layer
            self.conv2b = conv_layer(mid_chs, mid_chs, 3, stride=1, dilation=dilation, group=group)
        else:
            self.act2b = None
            self.conv2b = None
        if reg and attn_layer is not None:
            self.attn = attn_layer(mid_chs)  # RegNet blocks apply attn btw conv2 & 3
        else:
            self.attn = None
        self.act3 = act_layer
        self.conv3 = conv_layer(mid_chs, out_chs, 1, gain_init=1. if skipinit else 0.)
        if not reg and attn_layer is not None:
            self.attn_last = attn_layer(out_chs)  # ResNet blocks apply attn after conv3
        else:
            self.attn_last = None
        self.drop_path = DropPath2D(drop_path_rate) if drop_path_rate > 0 else None
        self.skipinit_gain = Parameter(Tensor(0.), mstype.float32) if skipinit else None

    def construct(self, x):
        """NormFreeBlock Construct"""
        out = self.act1(x) * self.beta
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)
        # residual branch
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))

        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        if not self.drop_path is None:
            out = self.drop_path(out)
        if self.skipinit_gain is not None:
            out *= self.skipinit_gain
        out = out * self.alpha + shortcut
        return out


def create_stem(in_chs, out_chs, stem_type='', conv_layer=None, act_layer=None):
    """create_stem for NFNet"""
    stem_stride = 2
    stem_feature = dict(num_chs=out_chs, reduction=2, module='stem.conv')
    stem = OrderedDict()
    assert stem_type in ('', 'deep', 'deep_tiered', 'deep_quad', '3x3', '7x7', 'deep_pool', '3x3_pool', '7x7_pool')
    if 'deep' in stem_type:
        if 'quad' in stem_type:
            # 4 deep conv stack as in NFNet-F models
            assert 'pool' not in stem_type
            stem_chs = (out_chs // 8, out_chs // 4, out_chs // 2, out_chs)
            strides = (2, 1, 1, 2)
            stem_stride = 4
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv3')
        else:
            if 'tiered' in stem_type:
                stem_chs = (3 * out_chs // 8, out_chs // 2, out_chs)  # 'T' resnets in resnet.py
            else:
                stem_chs = (out_chs // 2, out_chs // 2, out_chs)  # 'D' ResNets
            strides = (2, 1, 1)
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv2')
        last_idx = len(stem_chs) - 1
        for i, (c, s) in enumerate(zip(stem_chs, strides)):
            stem[f'conv{i + 1}'] = conv_layer(in_chs, c, kernel_size=3, stride=s)
            if i != last_idx:
                stem[f'act{i + 2}'] = act_layer
            in_chs = c
    elif '3x3' in stem_type:
        # 3x3 stem conv as in RegNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=3, stride=2)
    else:
        # 7x7 stem conv as in ResNet
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)

    if 'pool' in stem_type:
        stem['pool'] = nn.MaxPool2d(3, stride=2, pad_mode="same")
        stem_stride = 4

    return nn.SequentialCell(stem), stem_stride, stem_feature


# from https://github.com/deepmind/deepmind-research/tree/master/nfnets
_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


class NormFreeNet(nn.Cell):
    """ Normalization-Free Network

    As described in :
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    and
    `High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171

    This model aims to cover both the NFRegNet-Bx models as detailed in the paper's code snippets and
    the (preact) ResNet models described earlier in the paper.

    There are a few differences:
        * channels are rounded to be divisible by 8 by default (keep tensor core kernels happy),
            this changes channel dim and param counts slightly from the paper models
        * activation correcting gamma constants are moved into the ScaledStdConv as it has less performance
            impact in PyTorch when done with the weight scaling there. This likely wasn't a concern in the JAX impl.
        * a config option `gamma_in_act` can be enabled to not apply gamma in StdConv as described above, but
            apply it in each activation. This is slightly slower, numerically different, but matches official impl.
        * skipinit is disabled by default, it seems to have a rather drastic impact on GPU memory use and throughput
            for what it is/does. Approx 8-10% throughput loss.
    """

    def __init__(self, cfg, num_classes=1000, in_channel=3, global_pool='avg', output_stride=32, drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert cfg.act_layer in _nonlin_gamma, f"Please add non-linearity constants for activation ({cfg.act_layer})."
        conv_layer = ScaledStdConv2dSame
        if cfg.gamma_in_act:
            act_layer = act_with_gamma(cfg.act_layer, gamma=_nonlin_gamma[cfg.act_layer])
            conv_layer = partial(conv_layer, eps=cfg.std_conv_eps)
        else:
            act_layer = get_act_layer(cfg.act_layer)
            conv_layer = partial(conv_layer, gamma=_nonlin_gamma[cfg.act_layer], eps=cfg.std_conv_eps)
        attn_layer = partial(SEModule, **cfg.attn_kwargs) if cfg.attn_layer else None

        stem_chs = make_divisible((cfg.stem_chs or cfg.channels[0]) * cfg.width_factor, cfg.ch_div)
        self.stem, stem_stride, stem_feat = create_stem(in_channel, stem_chs, cfg.stem_type, conv_layer=conv_layer,
                                                        act_layer=act_layer)

        self.feature_info = [stem_feat]
        _drop_path_rates = [x for x in np.linspace(0, drop_path_rate, sum(cfg.depths))]
        drop_path_rates = []
        for index in range(len(cfg.depths)):
            drop_path_rates.append(_drop_path_rates[sum(cfg.depths[:index]):sum(cfg.depths[:index + 1])])

        prev_chs = stem_chs
        net_stride = stem_stride
        dilation = 1
        expected_var = 1.0
        stages = []
        for stage_idx, stage_depth in enumerate(cfg.depths):
            stride = 1 if stage_idx == 0 and stem_stride > 2 else 2
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2

            blocks = []
            for block_idx in range(stage_depth):
                first_block = block_idx == 0 and stage_idx == 0
                out_chs = make_divisible(cfg.channels[stage_idx] * cfg.width_factor, cfg.ch_div)
                blocks += [NormFreeBlock(
                    in_chs=prev_chs, out_chs=out_chs,
                    alpha=cfg.alpha,
                    beta=1. / expected_var ** 0.5,
                    stride=stride if block_idx == 0 else 1,
                    dilation=dilation,
                    first_dilation=first_dilation,
                    group_size=cfg.group_size,
                    bottle_ratio=1. if cfg.reg and first_block else cfg.bottle_ratio,
                    ch_div=cfg.ch_div,
                    reg=cfg.reg,
                    extra_conv=cfg.extra_conv,
                    skipinit=cfg.skipinit,
                    attn_layer=attn_layer,
                    attn_gain=cfg.attn_gain,
                    act_layer=act_layer,
                    conv_layer=conv_layer,
                    drop_path_rate=drop_path_rates[stage_idx][block_idx],
                )]
                if block_idx == 0:
                    expected_var = 1.  # expected var is reset after first block of each stage
                expected_var += cfg.alpha ** 2  # Even if reset occurs, increment expected variance
                first_dilation = dilation
                prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')]
            stages += [nn.SequentialCell(blocks)]
        self.stages = nn.SequentialCell(stages)

        if cfg.num_features:
            # The paper NFRegNet models have an EfficientNet-like final head convolution.
            self.num_features = make_divisible(cfg.width_factor * cfg.num_features, cfg.ch_div)
            self.final_conv = conv_layer(prev_chs, self.num_features, 1)
            self.feature_info[-1] = dict(num_chs=self.num_features, reduction=net_stride, module=f'final_conv')
        else:
            self.num_features = prev_chs
            self.final_conv = None
        self.final_act = act_layer

        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.01),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.Conv2d):
                stddev = np.sqrt(1 / np.prod(cell.weight.shape[1:]))
                cell.weight.set_data(weight_init.initializer(RandomNormal(stddev),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def construct_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        if not self.final_conv is None:
            x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def nfnet_f0(**kwargs):
    """ NFNet-F0
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return NormFreeNet(cfg=_nfnet_cfg(depths=[1, 2, 6, 3]), **kwargs)


def dm_nfnet_f0(**kwargs):
    """ NFNet-F0
    `High-Performance Large-Scale Image Recognition Without Normalization`
        - https://arxiv.org/abs/2102.06171
    """
    return NormFreeNet(cfg=_nfnet_cfg(depths=[1, 2, 6, 3]), **kwargs)
