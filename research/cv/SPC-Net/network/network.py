# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P

import numpy as np
from network.resnet import resnet50
from network.style_repin import StyleRepresentation
from network.kaiming_normal import kaiming_normal


def l2_normalize(x):
    return ops.L2Normalize(axis=-1)(x)


def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for _, m in model.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                m.weight.set_data(Tensor(kaiming_normal(m.weight.data.shape, mode="fan_out", nonlinearity='relu')))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))


class _AtrousSpatialPyramidPoolingModule(nn.Cell):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=None):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        rates = (6, 12, 18) if rates is None else rates
        print("output_stride = {}".format(output_stride))
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(nn.SequentialCell(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
            nn.BatchNorm2d(reduction_dim), nn.ReLU()
        ))
        # other rates
        for r in rates:
            self.features.append(nn.SequentialCell(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, pad_mode='pad'),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU()
            ))
        self.features = nn.CellList(self.features)  # ModuleList --> CellList

        # img level features
        self.concat = P.Concat(axis=1)
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.SequentialCell(
            nn.Conv2d(in_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU())

    def construct(self, x):
        x_size = x.shape

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = P.ResizeBilinear((x_size[2], x_size[3]), True)(img_features)
        out = img_features

        for f in self.features:
            y = f(x)
            out = self.concat((out, y))
        return out


class DeepV3Plus(nn.Cell):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self,
                 num_classes=19,
                 criterion=None,
                 criterion_aux=None,
                 args=None):
        super(DeepV3Plus, self).__init__()
        self.num_classes = num_classes
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.batch_size = 4
        if args.num == 1:
            self.num_datasets = 10
        else:
            self.num_datasets = args.num
        self.gamma = 0.999
        self.gamma_rampup = False
        self.temperature = 0.1
        self.gamma_style = 0.9
        self.dis_mode = 'was'
        self.feadim_style1 = 64
        self.feadim_style2 = 256
        self.channel_wise = False
        self.feature_dim = 256
        self.wt_layer = None

        channel_3rd = 256
        final_channel = 2048

        resnet = resnet50(wt_layer=self.wt_layer)
        resnet.layer0 = nn.SequentialCell(resnet.conv1, resnet.bn1, resnet.relu, resnet.pad, resnet.maxpool)

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        os = 16

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256, output_stride=os)
        self.concat = P.Concat(axis=1)

        self.bot_fine = nn.SequentialCell(
            nn.Conv2d(channel_3rd, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU())

        self.bot_aspp = nn.SequentialCell(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.final1 = nn.SequentialCell(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.projection = nn.SequentialCell(
            nn.Conv2d(256, self.feature_dim, kernel_size=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, has_bias=True))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.projection)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        self.three_input_layer = False

        # define prototype
        prototypes_init = Tensor(np.zeros((self.num_classes, self.num_datasets, self.feature_dim)), ms.float32)
        self.prototypes = ms.Parameter(prototypes_init, requires_grad=True)
        self.feat_norm = nn.LayerNorm([self.feature_dim], epsilon=1e-5)
        self.mask_norm = nn.LayerNorm([self.num_classes], epsilon=1e-5)

        self.style_adain1 = StyleRepresentation(
            num_prototype=self.num_datasets,
            channel_size=self.feadim_style1,
            batch_size=self.batch_size,
            gamma=self.gamma_style,
            dis_mode=self.dis_mode,
            channel_wise=self.channel_wise
        )
        self.style_adain2 = StyleRepresentation(
            num_prototype=self.num_datasets,
            channel_size=self.feadim_style2,
            batch_size=self.batch_size,
            gamma=self.gamma_style,
            dis_mode=self.dis_mode,
            channel_wise=self.channel_wise
        )

    def encoder(self, x):
        w_arr = []
        x = self.layer0(x)
        x = self.style_adain1(x)
        x_tuple = self.layer1([x, w_arr])  # [N, 256, 192, 192]
        low_level = x_tuple[0]  # [N, 256, 192, 192]
        x_tuple[0] = self.style_adain2(x_tuple[0])
        x_tuple = self.layer2(x_tuple)  # [N, 512, 96, 96]
        x_tuple = self.layer3(x_tuple)  # [N, 1024, 48, 48]
        aux_out = x_tuple[0]  # [N, 1024, 48, 48]
        x_tuple = self.layer4(x_tuple)  # [N, 2048, 48, 48]
        x = x_tuple[0]  # [N, 2048, 48, 48]
        w_arr = x_tuple[1]  # []
        return x, aux_out, low_level

    def decoder(self, x, low_level):
        x = self.aspp(x)  # [N, 1280, 48, 48]
        dec0_up = self.bot_aspp(x)  # [N, 256, 48, 48]
        dec0_fine = self.bot_fine(low_level)  # [N, 48, 192, 192]
        low_level_size = low_level.shape
        dec0_up = P.ResizeBilinear((low_level_size[2], low_level_size[3]), True)(dec0_up)
        dec0 = self.concat((dec0_fine, dec0_up))  # [N, 304, 192, 192]
        dec1 = self.final1(dec0)  # [N, 256, 192, 192]
        main_out = self.projection(dec1)
        main_out = ops.L2Normalize(axis=1)(main_out)
        return main_out

    def construct(self, x):
        x_size = x.shape  # [N, 3, 768, 768]
        # encoder
        x, _, low_level = self.encoder(x)
        # decoder
        _fea_out = self.decoder(x, low_level)   # [B, C, H, W], [8, 256, 192, 192]
        b, _, h, w = _fea_out.shape
        k = self.num_classes
        fea_out = ops.transpose(_fea_out, (0, 2, 3, 1))
        fea_out = fea_out.view(-1, fea_out.shape[-1])
        fea_out = self.feat_norm(fea_out)
        fea_out = l2_normalize(fea_out)
        self.prototypes = l2_normalize(self.prototypes)
        masks = ops.Einsum('nd,kmd->nmk')((fea_out, self.prototypes))
        _, main_out = ops.ArgMaxWithValue(axis=1)(masks)
        main_out = self.mask_norm(main_out)
        main_out = main_out.view(b, h, w, k)
        main_out = ops.transpose(main_out, (0, 3, 1, 2))
        main_out = P.ResizeBilinear((x_size[2], x_size[3]), True)(main_out)
        return main_out


def deep_r50v3plusd(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, criterion=criterion, criterion_aux=criterion_aux, args=args)


def load_pretrained_model(net, loaded_dict, multi_device=False):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in loaded_dict.items():
        if multi_device:
            name = k
        else:
            name = k[7:] # remove module.
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net
