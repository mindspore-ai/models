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

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import initializer as init
from src.models.backbones import SUPPORTED_BACKBONES

# ------------------------------------------------------------------------------
#  MODNet Basic Modules
# ------------------------------------------------------------------------------

class IBNorm(nn.Cell):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def construct(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...])
        # no learnable parameters
        in_x = self.inorm(x[:, self.bnorm_channels:, ...])

        return ops.Concat(axis=1)((bn_x, in_x))



class Conv2dIBNormRelu(nn.Cell):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      pad_mode='pad', group=groups, has_bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU())

        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)

class AdaptiveAvgPool2d(nn.Cell):
    """AdaptiveAvgPool2d"""
    def __init__(self):
        """rcan"""
        super().__init__()
        self.ReduceMean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        """rcan"""
        return self.ReduceMean(x, (2, 3))

class SEBlock(nn.Cell):  # channel-wise attention mechanisms
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = AdaptiveAvgPool2d() # Currently, ReduceMean is faster.
        # self.pool = nn.AdaptiveAvgPool2d(1) # the speed of nn.AdaptiveAvgPool2d will be optimized.
        self.fc = nn.SequentialCell(
            nn.Dense(in_channels, int(in_channels // reduction), has_bias=False),
            nn.ReLU(),
            nn.Dense(int(in_channels // reduction), out_channels, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


# ------------------------------------------------------------------------------
#  MODNet Branches
# ------------------------------------------------------------------------------

class LRBranch(nn.Cell):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()

        self.backbone = backbone
        enc_channels = self.backbone.enc_channels

        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                        with_relu=False)
        self.ResizeBilinear = nn.ResizeBilinear()


    def construct(self, img, inference):
        enc_features = self.backbone(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = self.ResizeBilinear(enc32x, scale_factor=2, align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = self.ResizeBilinear(lr16x, scale_factor=2, align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = ops.Sigmoid()(lr)

        return pred_semantic, lr8x, enc2x, enc4x  # lr8x is SI


class HRBranch(nn.Cell):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.SequentialCell(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.SequentialCell(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, img, enc2x, enc4x, lr8x, inference):
        size_img = img.shape[-2:]
        size_img2x = [x//2 for x in size_img]
        size_img4x = [x//4 for x in size_img]
        img2x = self.resize_bilinear(img, size=size_img2x, align_corners=False)
        img4x = self.resize_bilinear(img, size=size_img4x, align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(ops.Concat(axis=1)((img2x, enc2x)))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(ops.Concat(axis=1)((hr4x, enc4x)))

        lr4x = self.resize_bilinear(lr8x, scale_factor=2, align_corners=False)
        hr4x = self.conv_hr4x(ops.Concat(axis=1)((hr4x, lr4x, img4x)))

        hr2x = self.resize_bilinear(hr4x, scale_factor=2, align_corners=False)
        hr2x = self.conv_hr2x(ops.Concat(axis=1)((hr2x, enc2x)))

        pred_detail = None
        if not inference:
            hr = self.resize_bilinear(hr2x, scale_factor=2, align_corners=False)
            hr = self.conv_hr(ops.Concat(axis=1)((hr, img)))
            pred_detail = ops.Sigmoid()(hr)

        return pred_detail, hr2x


class FusionBranch(nn.Cell):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.SequentialCell(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, img, lr8x, hr2x):
        lr4x = self.resize_bilinear(lr8x, scale_factor=2, align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = self.resize_bilinear(lr4x, scale_factor=2, align_corners=False)

        f2x = self.conv_f2x(ops.Concat(axis=1)((lr2x, hr2x)))
        f = self.resize_bilinear(f2x, scale_factor=2, align_corners=False)
        f = self.conv_f(ops.Concat(axis=1)((f, img)))
        pred_matte = ops.Sigmoid()(f)

        return pred_matte


# ------------------------------------------------------------------------------
#  MODNet
# ------------------------------------------------------------------------------

class MODNet(nn.Cell):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained
        backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(backbone)
        self.hr_branch = HRBranch(self.hr_channels, backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, backbone.enc_channels)

        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
                self._init_conv(cell)
            elif isinstance(cell, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                self._init_norm(cell)
        if self.backbone_pretrained:
            self.lr_branch.backbone.load_pretrained_ckpt()

    def construct(self, img, inference=False):

        pred_semantic, lr8x, enc2x, enc4x = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte

    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        conv.weight.set_data(init.initializer(
            init.HeUniform(negative_slope=0, mode='fan_in', nonlinearity='leaky_relu'), conv.weight.shape))
        if conv.bias is not None:
            conv.bias.set_data(init.initializer(0, conv.bias.shape))

    def _init_norm(self, norm):
        norm.gamma.set_data(init.initializer('ones', norm.gamma.shape))
        norm.beta.set_data(init.initializer('zeros', norm.beta.shape))
