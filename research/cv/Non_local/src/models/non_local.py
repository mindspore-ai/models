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

# This file was copied from project [feiyunzhang][i3d-non-local-pytorch]

import math
import mindspore
import mindspore.nn as nn
from mindspore import Parameter
import mindspore.numpy as np
from mindspore import ops
from mindspore.common.initializer import HeNormal
from mindspore import load_checkpoint, load_param_into_net
from mindspore import dtype as mstype
from mindspore.ops import constexpr


class NLBlockND(nn.Cell):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specified reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension
        self.batmatmul = mindspore.ops.BatchMatMul()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.cast = mindspore.ops.Cast()

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            self.max_pool_layer = mindspore.ops.MaxPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            self.max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, has_bias=True,
                         weight_init=HeNormal(mode='fan_out'))

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.out = nn.SequentialCell(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, has_bias=True,
                        weight_init=HeNormal(mode='fan_out')),
                bn(self.in_channels, gamma_init="zeros", beta_init="zeros")
                # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            )
        else:
            self.out = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1,
                               has_bias=True, weight_init="zeros", bias_init="zeros")
            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                                 has_bias=True, weight_init=HeNormal(mode='fan_out'))
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                               has_bias=True, weight_init=HeNormal(mode='fan_out'))

        if self.mode == "concatenate":
            self.W_f = nn.SequentialCell(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1,
                          weight_init=HeNormal(mode='fan_out')),
                nn.ReLU()
            )

    def construct(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x.shape[0]

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        mp = self.max_pool_layer(x)
        g_x = self.g(mp).view(batch_size, self.inter_channels, -1).transpose(0, 2, 1)

        f = 0
        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = mindspore.ops.transpose(theta_x, (0, 2, 1))
            f = self.batmatmul(theta_x, phi_x)
        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(mp).view(batch_size, self.inter_channels, -1)
            theta_x = mindspore.ops.transpose(theta_x, (0, 2, 1))
            phi_x = self.cast(phi_x, mstype.float16)
            theta_x = self.cast(theta_x, mstype.float16)
            f = self.batmatmul(theta_x, phi_x)
            f = self.cast(f, mstype.float32)
            f = f * (self.inter_channels ** -.5)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.shape[2]
            w = phi_x.shape[3]
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat_ops = mindspore.ops.Concat(axis=1)
            concat = concat_ops([theta_x, phi_x])
            f = self.W_f(concat)
            f = f.view(f.shape[0], f.shape[2], f.shape[3])

        f_div_C = 0
        if self.mode == "gaussian" or self.mode == "embedded":
            soft_max = mindspore.ops.Softmax()
            f_div_C = soft_max(f)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.shape[-1]  # number of position in x
            f_div_C = f / N
        f_div_C = self.cast(f_div_C, mstype.float16)
        g_x = self.cast(g_x, mstype.float16)
        y = self.batmatmul(f_div_C, g_x)
        y = self.cast(y, mstype.float32)
        y = y.transpose(0, 2, 1).ravel()

        y = y.view(batch_size, self.inter_channels, x.shape[2], x.shape[3], x.shape[4])

        out = self.out(y)
        # residual connection
        z = out + x

        return z


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, time_kernel=1, space_stride=1, downsample=None, non_local=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(time_kernel, 1, 1),
                               padding=(int((time_kernel - 1) / 2), int((time_kernel - 1) / 2), 0, 0, 0, 0),
                               pad_mode="pad",
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, space_stride, space_stride),
                               pad_mode="pad", padding=(0, 0, 1, 1, 1, 1),
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1, 1, 1),
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.nl = NLBlockND(in_channels=planes * 4, dimension=3) if non_local else None

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl.construct(out)

        return out


class C2DResNet(nn.Cell):
    """C2D with ResNet backbone.
    The only operation involving the temporal domain are the pooling layer after the second residual block.
    For more details of the structure, refer to Table 1 from the paper.
    Padding was added accordingly to match the correct dimensionality.
    """

    def __init__(self, block, layers, frame_num=32, num_classes=400):
        self.inplanes = 64
        super(C2DResNet, self).__init__()

        # first convolution operation has essentially 2D kernels
        # output: 64 x 16 x 112 x 112
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=2, pad_mode="pad", padding=(0, 0, 3, 3, 3, 3),
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()

        # output: 64 x 8 x 56 x 56
        self.pool1 = mindspore.ops.MaxPool3D(kernel_size=3, strides=2, pad_mode="same")

        # output: 256 x 8 x 56 x 56
        self.layer1 = self._make_layer(block, 64, layers[0], space_stride=1, d_padding=0)

        # pooling on temporal domain
        # output: 256 x 4 x 56 x 56
        self.pool_t = mindspore.ops.MaxPool3D(kernel_size=(3, 1, 1), strides=(2, 1, 1), pad_mode="same")

        # output: 512 x 4 x 28 x 28
        self.layer2 = self._make_layer(block, 128, layers[1], space_stride=2, non_local=True)

        # add one non-local block here
        # output: 1024 x 4 x 14 x 14
        self.layer3 = self._make_layer(block, 256, layers[2], space_stride=2, non_local=True)

        # output: 2048 x 4 x 7 x 7
        self.layer4 = self._make_layer(block, 512, layers[3], space_stride=2)

        # output: 2048 x 1
        self.avgpool = mindspore.ops.AvgPool3D(kernel_size=((int(frame_num / 8), 7, 7)))
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, space_stride=1, d_padding=0, non_local=False):
        downsample = nn.SequentialCell(
            nn.Conv3d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=(1, space_stride, space_stride), padding=d_padding, has_bias=False,
                      weight_init=HeNormal(mode='fan_out')),
            nn.BatchNorm3d(planes * block.expansion)
        )

        layers = []
        time_kernel = 1
        layers.append(block(self.inplanes, planes, time_kernel, space_stride, downsample))
        self.inplanes = planes * block.expansion

        if not non_local:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        elif blocks in (4, 6):
            for i in range(1, blocks):
                if i % 2 == 1:
                    layers.append(block(self.inplanes, planes, non_local=True))
                else:
                    layers.append(block(self.inplanes, planes))
        elif blocks == 23:
            for i in range(1, blocks):
                if i % 7 == 6:
                    layers.append(block(self.inplanes, planes, non_local=True))
                else:
                    layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.pool_t(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_time, kernel_width, kernel_height = inp_shape[2], inp_shape[3], inp_shape[4]
    if isinstance(output_size, int):
        kernel_time = math.ceil(kernel_time / output_size)
        kernel_width = math.ceil(kernel_width / output_size)
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, (list, tuple)):
        kernel_time = math.ceil(kernel_width / output_size[0])
        kernel_width = math.ceil(kernel_width / output_size[1])
        kernel_height = math.ceil(kernel_height / output_size[2])
    return (kernel_time, kernel_width, kernel_height)


class AdaptiveAvgPool3d(nn.Cell):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return mindspore.ops.AvgPool3D(kernel_size)(x)


class I3DResNet(nn.Cell):
    """I3D with ResNet backbone.
    """

    def __init__(self, block, layers, num_classes=400):
        self.inplanes = 64
        super(I3DResNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=2, padding=(2, 2, 3, 3, 3, 3), pad_mode="pad",
                               has_bias=False, weight_init=HeNormal(mode='fan_out'))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.stack = ops.Stack(1)

        self.maxpool1 = mindspore.ops.MaxPool3D(kernel_size=3, strides=2, pad_mode="pad", pad_list=1)

        self.layer1 = self._make_layer_inflat(block, 64, layers[0], time_kernel=[3, 3, 3])

        self.maxpool2 = mindspore.ops.MaxPool3D(kernel_size=(3, 1, 1), strides=(2, 1, 1), pad_mode="pad",
                                                pad_list=(1, 1, 0, 0, 0, 0))

        self.layer2 = self._make_layer_inflat(block, 128, layers[1], time_kernel=[3, 1, 3, 1], space_stride=2,
                                              non_local=True)

        self.layer3 = self._make_layer_inflat(block, 256, layers[2], time_kernel=[3, 1, 3, 1, 3, 1], space_stride=2,
                                              non_local=True)

        self.layer4 = self._make_layer_inflat(block, 512, layers[3], time_kernel=[1, 3, 1], space_stride=2)
        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.avgdrop = nn.Dropout(p=0.5)
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer_inflat(self, block, planes, blocks, time_kernel, space_stride=1, non_local=False):
        downsample = None
        if space_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1, 1), stride=(1, space_stride, space_stride), has_bias=False,
                          weight_init=HeNormal(mode='fan_out')),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, time_kernel[0], space_stride, downsample))
        self.inplanes = planes * block.expansion

        if not non_local:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, time_kernel[i]))
        elif blocks in (4, 6):
            for i in range(1, blocks):
                if i % 2 == 1:
                    layers.append(block(self.inplanes, planes, time_kernel[i], non_local=True))
                else:
                    layers.append(block(self.inplanes, planes, time_kernel[i], non_local=False))
        elif blocks == 23:
            for i in range(1, blocks):
                if i % 2 == 1:
                    time_kernel[i] = 3
                else:
                    time_kernel[i] = 1
                if i % 7 == 6:
                    layers.append(block(self.inplanes, planes, time_kernel[i]))
                else:
                    layers.append(block(self.inplanes, planes, time_kernel[i]))

        return nn.SequentialCell(*layers)

    def get_optim_policies(self):
        first_conv_weight = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        for m in self.trainable_params():
            if 'conv' in m.name:
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(m)
                else:
                    if 'weight' in m.name:
                        normal_weight.append(m)
                    elif 'bias' in m.name:
                        normal_bias.append(m)
            elif 'fc' in m.name:
                if 'weight' in m.name:
                    normal_weight.append(m)
                elif 'bias' in m.name:
                    normal_bias.append(m)
            elif 'bn' in m.name:  # enable BN
                bn.append(m)
            elif 'downsample' in m.name:
                if 'weight' in m.name:
                    normal_weight.append(m)
                elif 'bias' in m.name:
                    normal_bias.append(m)
        return [
            {'params': first_conv_weight},
            {'params': normal_weight},
            {'params': normal_bias, 'weight_decay': 0},
            {'params': bn, 'weight_decay': 0},
        ]

    def construct_single(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x, (2, 3, 4))
        x = self.avgdrop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x

    def construct_multi(self, x):
        clip_preds = []
        for clip_idx in range(x.shape[1]):  # B, 10, 3, 3, 32, 256, 256
            spatial_crops = []
            for crop_idx in range(x.shape[2]):
                clip = x[:, clip_idx, crop_idx]
                clip = self.construct_single(clip)
                spatial_crops.append(clip)
            spatial_crops = self.stack(spatial_crops).mean(1)  # (B, 400)
            clip_preds.append(spatial_crops)
        clip_preds = self.stack(clip_preds).mean(1)  # (B, 400)
        return clip_preds

    def construct(self, x):
        pred = None
        if x.ndim == 5:
            pred = self.construct_single(x)
        elif x.ndim == 7:
            pred = self.construct_multi(x)
        # print(ops.Argmax()(pred))

        return pred


def inflat_weights(pretrained_dict_2d, model_3d):
    model_dict_3d = model_3d.parameters_dict()
    for key, weight_2d in pretrained_dict_2d.items():
        if key in model_dict_3d:
            if 'conv' in key:
                time_kernel_size = model_dict_3d[key].shape[2]
                if 'weight' in key:
                    expand_dims = ops.ExpandDims()
                    weight_3d = expand_dims(weight_2d, 2)
                    weight_3d = np.tile(weight_3d, (1, 1, time_kernel_size, 1, 1))
                    weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = Parameter(weight_3d)
                elif 'bias' in key:
                    model_dict_3d[key] = Parameter(weight_2d)
            elif 'bn' in key:
                model_dict_3d[key] = Parameter(weight_2d)
            elif 'downsample' in key:
                if '0.weight' in key:
                    time_kernel_size = model_dict_3d[key].shape[2]
                    expand_dims = ops.ExpandDims()
                    weight_3d = expand_dims(weight_2d, 2)
                    weight_3d = np.tile(weight_3d, (1, 1, time_kernel_size, 1, 1))
                    weight_3d = weight_3d / time_kernel_size
                    model_dict_3d[key] = Parameter(weight_3d)
                else:
                    model_dict_3d[key] = Parameter(weight_2d)

    load_param_into_net(model_3d, model_dict_3d)
    return model_3d


def C2DResNet50(**kwargs):
    """Constructs a C2D ResNet-50 model.
    """
    model = C2DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def C2DResNet101(**kwargs):
    """Constructs a C2D ResNet-101 model.
    """
    model = C2DResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def I3DResNet50(**kwargs):
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    pretrained_dict_2d = load_checkpoint(kwargs['pretrained_ckpt'])
    model = inflat_weights(pretrained_dict_2d, model)
    return model


def I3DResNet101(**kwargs):
    model = I3DResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
