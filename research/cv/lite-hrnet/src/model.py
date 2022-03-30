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

from mindspore import ops
from mindspore import nn
from mindspore.common import initializer

BN_MOMENTUM = 0.9


def channel_shuffle(x, groups):
    """Channel Shuffle operation.
    This function enables cross-group information flow for multiple groups
    convolution layers.
    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.
    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(0, 2, 1, 3, 4)
    x = x.view(batch_size, -1, height, width)

    return x


class ConvModule(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 dilation=1,
                 conv_cfg=None,
                 act_cfg=None,
                 norm_cfg=None):
        super().__init__()
        if norm_cfg is not None:
            bias = False
        else:
            bias = True
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              pad_mode='pad',
                              padding=padding,
                              group=groups,
                              dilation=dilation,
                              has_bias=bias)

        self.norm = None
        if norm_cfg:
            if norm_cfg['type'] == 'BN':
                self.norm = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            else:
                raise TypeError('Unknown normalization layer')

        self.act = None
        if act_cfg:
            if act_cfg['type'] == 'ReLU':
                self.act = nn.ReLU()
            elif act_cfg['type'] == 'Sigmoid':
                self.act = nn.Sigmoid()
            else:
                raise TypeError('Unknown activation layer')

    def construct(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


def build_conv_layer(cfg,
                     in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        pad_mode='pad',
        padding=padding,
        group=groups,
        dilation=dilation,
        has_bias=bias)

def build_norm_layer(cfg, channels):
    if cfg['type'] == 'BN':
        norm = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
    else:
        raise TypeError('Unknown normalization layer')
    return norm


class DynamicUpsample(nn.Cell):
    def __init__(self, scale_factor=1, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def construct(self, x):
        shape = x.shape[-2:]
        if self.mode == 'nearest':
            operation = ops.ResizeNearestNeighbor((shape[0]*self.scale_factor, shape[1]*self.scale_factor))(x)
        else:
            operation = nn.ResizeBilinear()(x, size=(shape[0]*self.scale_factor, shape[1]*self.scale_factor))
        return operation


class IdentityCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dummy_unique_attribute = True

    def construct(self, x):
        return x



class DepthwiseSeparableConvModule(nn.Cell):
    """Depthwise separable convolution module.
    See https://arxiv.org/pdf/1704.04861.pdf for details.
    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=None,
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs)

    def construct(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SpatialWeighting(nn.Cell):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.global_avgpool = ops.ReduceMean(keep_dims=True)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def construct(self, x):
        out = self.global_avgpool(x, (2, 3))
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Cell):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        mini_size = (x[-1]).shape[-2:]
        out = []
        for i in range(len(self.channels) - 1):
            pooling = ops.AdaptiveAvgPool2D(mini_size)
            out.append(pooling(x[i]))
        out.append(x[-1])
        out = self.concat(out)
        out = self.conv1(out)
        out = self.conv2(out)

        out_splitted = [out[:, :self.channels[0], :, :]]
        current_index = self.channels[0]
        for i in range(1, len(self.channels) - 1):
            out_splitted.append(out[:, current_index:current_index + self.channels[i], :, :])
            current_index = current_index + self.channels[i]
        out_splitted.append(out[:, -self.channels[-1]:, :, :])
        out = out_splitted

        result = []
        for i in range(len(x)):
            result.append(x[i] * ops.ResizeNearestNeighbor(x[i].shape[-2:])(out[i]))
        return result


class ConditionalChannelWeighting(nn.Cell):

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]
        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.depthwise_convs = nn.CellList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.CellList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])
        self.chunk = ops.Split(axis=1, output_num=2)

    def construct(self, x):
        x_new = []
        for s in x:
            x_new.append(self.chunk(s))
        x = x_new

        x1 = []
        x2 = []
        for s in x:
            x1.append(s[0])
            x2.append(s[1])

        x2 = self.cross_resolution_weighting(x2)

        x2_new = []
        for s, dw in zip(x2, self.depthwise_convs):
            x2_new.append(dw(s))
        x2 = x2_new

        x2_new = []
        for s, sw in zip(x2, self.spatial_weighting):
            x2_new.append(sw(s))
        x2 = x2_new

        out_new = []
        for s1, s2 in zip(x1, x2):
            out_new.append(ops.Concat(axis=1)([s1, s2]))
        out = out_new

        out_new = []
        for s in out:
            out_new.append(channel_shuffle(s, 2))
        out = out_new

        return out


class Stem(nn.Cell):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        if norm_cfg is not None:
            self.norm_cfg = norm_cfg
        else:
            self.norm_cfg = dict(type='BN')
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.SequentialCell(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.chunk = ops.Split(axis=1, output_num=2)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        x = self.conv1(x)
        x1, x2 = self.chunk(x)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        x1 = self.branch1(x1)
        out = self.concat([x1, x2])

        out = channel_shuffle(out, 2)

        return out


class IterativeHead(nn.Cell):

    def __init__(self, in_channels, conv_cfg=None, norm_cfg=None):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]
        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.CellList(projects)

    def construct(self, x):
        x_new = [0] * len(x)
        for i, s in enumerate(x):
            x_new[len(x)-i-1] = s
        x = x_new

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = ops.ResizeBilinear(s.shape[-2:], align_corners=True)(last_x)
                s = ops.Add()(s, last_x)

            s = self.projects[i](s)
            y.append(s)
            last_x = s

        y_new = [0] * len(y)
        for i, s in enumerate(y):
            y_new[len(y)-i-1] = s
        return y_new


class LiteHRModule(nn.Cell):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True,
            conv_cfg=None,
            norm_cfg=None,
            with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        if norm_cfg is not None:
            self.norm_cfg = norm_cfg
        else:
            self.norm_cfg = dict(type='BN')
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        else: raise TypeError('Only LITE module type implemented')

        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))

        return nn.SequentialCell(layers)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.SequentialCell(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i]),
                            DynamicUpsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(IdentityCell())
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.SequentialCell(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j]),
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])))
                        else:
                            conv_downsamples.append(
                                nn.SequentialCell(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j]),
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j]),
                                    nn.ReLU()))
                    fuse_layer.append(nn.SequentialCell(conv_downsamples))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def construct(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        out = self.layers(x)

        if self.with_fuse:
            out_fuse = []
            y = 0
            for i in range(len(self.fuse_layers)):
                if i == 1:
                    new_out = [y]
                    for k in range(1, len(self.fuse_layers)):
                        new_out.append(out[k])
                    out = new_out
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y = ops.Add()(y, out[j])
                    else:
                        y = ops.Add()(y, self.fuse_layers[i][j](out[j]))
                out_fuse.append(self.relu(y))
            out = out_fuse
        return out


class LiteHRNet(nn.Cell):
    """Lite-HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    """

    def __init__(self,
                 extra,
                 head_extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super().__init__()
        self.extra = extra
        self.head_extra = head_extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if norm_cfg is not None:
            self.norm_cfg = norm_cfg
        else:
            self.norm_cfg = dict(type='BN')
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.print = ops.Print()

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            expand_ratio=self.extra['stem']['expand_ratio'],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]
        self.attr_dict = {'transition': [],
                          'stage': [],
                          'flag': []}
        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            transition, flag = self._make_transition_layer(num_channels_last, num_channels)
            self.attr_dict['transition'].append(transition)
            self.attr_dict['flag'].append(flag)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True)
            self.attr_dict['stage'].append(stage)

        self.with_head = self.extra['with_head']
        if self.with_head:
            self.head_layer = IterativeHead(
                in_channels=num_channels_last,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )

        self.transition00 = self.attr_dict['transition'][0][0]
        self.transition01 = self.attr_dict['transition'][0][1]
        self.transition12 = self.attr_dict['transition'][1][2]
        self.transition23 = self.attr_dict['transition'][2][3]

        self.stage0 = self.attr_dict['stage'][0]
        self.stage1 = self.attr_dict['stage'][1]
        self.stage2 = self.attr_dict['stage'][2]

        self.final_layer = ConvModule(self.head_extra['in_channels'],
                                      self.head_extra['out_channels'])

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        flag = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_pre_layer[i]),
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i]),
                            nn.ReLU()))
                    flag.append('ops')
                else:
                    transition_layers.append(IdentityCell())
                    flag.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.SequentialCell(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels),
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels),
                            nn.ReLU()))
                transition_layers.append(nn.SequentialCell(conv_downsamples))
                flag.append('ops')

        return nn.CellList(transition_layers), flag

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        module_type = stages_spec['module_type'][stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))
            in_channels = modules[-1].in_channels

        return nn.SequentialCell(modules), in_channels


    def construct(self, x):
        """Forward function."""
        x = self.stem(x)

        y_list = [x]

        x_list = []
        x_list.append(self.transition00(y_list[0]))
        x_list.append(self.transition01(y_list[0]))
        y_list = self.stage0(x_list)

        x_list = []
        x_list.append(y_list[0])
        x_list.append(y_list[1])
        x_list.append(self.transition12(y_list[1]))
        y_list = self.stage1([x_list[0], x_list[1], x_list[2]])

        x_list = []
        x_list.append(y_list[0])
        x_list.append(y_list[1])
        x_list.append(y_list[2])
        x_list.append(self.transition23(y_list[2]))
        y_list = self.stage2(x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)

        x = self.final_layer(x[0])

        return x


class LiteHRNetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, target, target_weight):
        output = self._backbone(img)
        return self._loss_fn(output, target, target_weight)


def get_posenet_model(cfg):
    """Create HRNet object, and initialize it by initializer or checkpoint."""
    backbone = LiteHRNet(cfg['backbone']['extra'], cfg['keypoint_head'])

    for _, cell in backbone.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=0.001),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer.initializer(0,
                                                           cell.bias.shape,
                                                           cell.bias.dtype))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(initializer.initializer(1,
                                                        cell.gamma.shape,
                                                        cell.gamma.dtype))
            cell.beta.set_data(initializer.initializer(0,
                                                       cell.beta.shape,
                                                       cell.beta.dtype))

    return backbone
