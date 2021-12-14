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
Some Commonly Used Block
"""

import functools
from mindspore import nn, ops


class ResnetBlock(nn.Cell):
    """ResnetBlock"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """build_conv_block"""
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="REFLECT")]
        elif padding_type == 'symmetric':
            conv_block += [nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC")]
        elif padding_type == 'constant':
            conv_block += [nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, has_bias=use_bias, pad_mode="pad"),
                       norm_layer(dim),
                       nn.ReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="REFLECT")]
        elif padding_type == 'symmetric':
            conv_block += [nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC")]
        elif padding_type == 'constant':
            conv_block += [nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, has_bias=use_bias, pad_mode="pad"),
                       norm_layer(dim)]

        return nn.SequentialCell(conv_block)

    def construct(self, x):
        out = x + self.conv_block(x)
        return out


class UnetSkipConnectionBlock(nn.Cell):
    """UnetSkipConnectionBlock"""
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, has_bias=use_bias, pad_mode="pad")
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.Conv2dTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, pad_mode="pad", has_bias=True)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            if submodule is not None:
                model = down + [submodule] + up
            else:
                model = down + up
        elif innermost:
            upconv = nn.Conv2dTranspose(inner_nc, outer_nc, has_bias=use_bias,
                                        kernel_size=4, stride=2,
                                        padding=1, pad_mode="pad")
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.Conv2dTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, has_bias=use_bias, pad_mode="pad")
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                if submodule is not None:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + up + [nn.Dropout(0.5)]
            else:
                if submodule is not None:
                    model = down + [submodule] + up
                else:
                    model = down + up

        self.model = nn.SequentialCell(model)

    def construct(self, x):
        if self.outermost:
            return self.model(x)
        op = ops.Concat(1)
        output = op((x, self.model(x)))
        return output
