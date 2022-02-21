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
"""
networks define
"""
import numpy as np

import mindspore
import mindspore.ops as ops
from mindspore import nn, Parameter, Tensor
from mindspore import dtype as mstype

from src.modelarts_utils.config import config


class UpSample(nn.Cell):
    def __init__(self, scale_factor, img_size):
        super(UpSample, self).__init__()
        self.upsample = ops.ResizeNearestNeighbor((img_size * scale_factor // 4, img_size * scale_factor // 4))

    def construct(self, x):
        return self.upsample(x)

class Dense(nn.Cell):
    """dense layer of float16 or float32"""
    def __init__(self, in_channels, out_channels, weight_init='uniform', has_bias=False, target_device='Ascend'):
        super(Dense, self).__init__()
        self.target_device = target_device
        if self.target_device == 'Ascend':
            self.dense = nn.Dense(in_channels=in_channels,
                                  out_channels=out_channels,
                                  weight_init=weight_init,
                                  has_bias=has_bias).to_float(mstype.float16)
        else:
            self.dense = nn.Dense(in_channels=in_channels,
                                  out_channels=out_channels,
                                  weight_init=weight_init,
                                  has_bias=has_bias)

    def construct(self, x):
        if self.target_device == 'Ascend':
            x = ops.cast(x, mstype.float16)
            out = self.dense(x)
            out = ops.cast(out, mstype.float32)
        else:
            out = self.dense(x)
        return out

class ResnetGenerator(nn.Cell):
    """resnet generator"""
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode="CONSTANT"),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1,
                                padding=0, has_bias=False, pad_mode="pad"),
                      nn.BatchNorm2d(ngf, affine=False),
                      nn.ReLU()]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2,
                                    padding=0, has_bias=False, pad_mode="pad"),
                          nn.BatchNorm2d(ngf * mult * 2, affine=False),
                          nn.ReLU()]

        # Down-Sampling Bottleneck
        # The DownBlock will downsample the original image size into (h/4, w/4)
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Substitutions for AdaptiveAvgPooling and AdaptiveMaxPooling
        self.ada_avg_pool = ops.ReduceMean(keep_dims=True)
        self.ada_max_pool = ops.ReduceMax(keep_dims=True)

        # Class Activation Map
        self.gap_fc = Dense(in_channels=ngf * mult,
                            out_channels=1,
                            weight_init='uniform',
                            has_bias=False,
                            target_device=config.device_target)
        self.gmp_fc = Dense(in_channels=ngf * mult,
                            out_channels=1,
                            weight_init='uniform',
                            has_bias=False,
                            target_device=config.device_target)

        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1,
                                 stride=1, has_bias=True, bias_init='uniform')
        #self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, has_bias=False, bias_init='uniform')
        self.relu = nn.ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [Dense(in_channels=ngf * mult,
                        out_channels=ngf * mult,
                        weight_init='uniform',
                        has_bias=False,
                        target_device=config.device_target),
                  nn.ReLU(),
                  Dense(in_channels=ngf * mult,
                        out_channels=ngf * mult,
                        weight_init='uniform',
                        has_bias=False,
                        target_device=config.device_target),
                  nn.ReLU()]
        else:
            FC = [Dense(in_channels=img_size // mult * img_size // mult * ngf * mult,
                        out_channels=ngf * mult,
                        weight_init='uniform',
                        has_bias=False,
                        target_device=config.device_target),
                  nn.ReLU(),
                  Dense(in_channels=ngf * mult,
                        out_channels=ngf * mult,
                        weight_init='uniform',
                        has_bias=False,
                        target_device=config.device_target),
                  nn.ReLU()]
        self.gamma = Dense(in_channels=ngf * mult,
                           out_channels=ngf * mult,
                           weight_init='uniform',
                           has_bias=False,
                           target_device=config.device_target)
        self.beta = Dense(in_channels=ngf * mult,
                          out_channels=ngf * mult,
                          weight_init='uniform',
                          has_bias=False,
                          target_device=config.device_target)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            scale_factor = 2**(i + 1)
            UpBlock2 += [UpSample(scale_factor=scale_factor, img_size=self.img_size),
                         nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1,
                                   padding=0, has_bias=False, pad_mode="pad"),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU()]

        UpBlock2 += [nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode="CONSTANT"),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1,
                               padding=0, has_bias=False, pad_mode="pad"),
                     nn.Tanh()]

        self.DownBlock = nn.SequentialCell(DownBlock)
        self.FC = nn.SequentialCell(FC)
        self.UpBlock2 = nn.SequentialCell(UpBlock2)
        self.expand_dims = ops.ExpandDims()

        # utils
        self.concat = ops.Concat(1)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)


    def construct(self, img):
        """construct"""
        x = self.DownBlock(img)

        gap = self.ada_avg_pool(x, (2, 3))
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = self.gap_fc.dense.weight
        gap_weight = self.expand_dims(gap_weight, 2)
        gap_weight = self.expand_dims(gap_weight, 3)
        gap = x * gap_weight

        gmp = self.ada_max_pool(x, (2, 3))
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = self.gmp_fc.dense.weight
        gmp_weight = self.expand_dims(gmp_weight, 2)
        gmp_weight = self.expand_dims(gmp_weight, 3)
        gmp = x * gmp_weight

        cam_logit = self.concat((gap_logit, gmp_logit))
        x = self.concat((gap, gmp))
        x = self.relu(self.conv1x1(x))
        heatmap = self.reduce_sum(x, 1)
        if self.light:
            x_ = self.ada_avg_pool(x, (2, 3))
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        x = self.UpBlock1_1(x, gamma, beta)
        x = self.UpBlock1_2(x, gamma, beta)
        x = self.UpBlock1_3(x, gamma, beta)
        x = self.UpBlock1_4(x, gamma, beta)

        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(nn.Cell):
    """resnet block"""
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 padding=0, has_bias=use_bias, pad_mode="pad"),
                       nn.BatchNorm2d(dim, affine=False),
                       nn.ReLU()]

        conv_block += [nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 padding=0, has_bias=use_bias, pad_mode="pad"),
                       nn.BatchNorm2d(dim, affine=False)]

        self.conv_block = nn.SequentialCell(conv_block)

    def construct(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetAdaILNBlock(nn.Cell):
    """resnet adaILN block"""
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                               padding=0, has_bias=use_bias, pad_mode="pad")
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                               padding=0, has_bias=use_bias, pad_mode="pad")
        self.norm2 = adaILN(dim)

    def construct(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Cell):
    """adaILN"""
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = mindspore.Parameter(Tensor(0.9*np.ones((1, num_features, 1, 1)), mindspore.float32))
        self.mean = ops.ReduceMean(keep_dims=True)
        self.expand_dims = ops.ExpandDims()
        self.sqrt = ops.Sqrt()

    def construct(self, feat, gamma, beta):
        """construct"""
        in_mean, in_var = self.mean(feat, (2, 3)), feat.var((2, 3), 0, True)
        out_in = (feat - in_mean) / self.sqrt(in_var + self.eps)
        ln_mean, ln_var = self.mean(feat, (1, 2, 3)), feat.var((1, 2, 3), 0, True)
        out_ln = (feat - ln_mean) / self.sqrt(ln_var + self.eps)
        expand = ops.BroadcastTo((feat.shape[0], -1, -1, -1))
        out = expand(self.rho) * out_in + (1 - expand(self.rho)) * out_ln
        gamma = self.expand_dims(gamma, 2)
        gamma = self.expand_dims(gamma, 3)
        beta = self.expand_dims(beta, 2)
        beta = self.expand_dims(beta, 3)
        out = out * gamma + beta

        return out

class ILN(nn.Cell):
    """ILN"""
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(Tensor(np.zeros((1, num_features, 1, 1)), mindspore.float32))
        self.gamma = Parameter(Tensor(np.ones((1, num_features, 1, 1)), mindspore.float32))
        self.beta = Parameter(Tensor(np.zeros((1, num_features, 1, 1)), mindspore.float32))
        self.mean = ops.ReduceMean(keep_dims=True)
        self.sqrt = ops.Sqrt()

    def construct(self, feat):
        in_mean, in_var = self.mean(feat, (2, 3)), feat.var((2, 3), 0, True)
        out_in = (feat - in_mean) / self.sqrt(in_var + self.eps)
        ln_mean, ln_var = self.mean(feat, (1, 2, 3)), feat.var((1, 2, 3), 0, True)
        out_ln = (feat - ln_mean) / self.sqrt(ln_var + self.eps)
        expand = ops.BroadcastTo((feat.shape[0], -1, -1, -1))
        out = expand(self.rho) * out_in + (1-expand(self.rho)) * out_ln
        out = out * expand(self.gamma) + expand(self.beta)

        return out

class Discriminator(nn.Cell):
    """discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2,
                           padding=0, has_bias=True, pad_mode="pad"),
                 nn.LeakyReLU(0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2,
                                padding=0, has_bias=True, pad_mode="pad"),
                      nn.LeakyReLU(0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1,
                            padding=0, has_bias=True, pad_mode="pad"),
                  nn.LeakyReLU(0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Dense(ndf * mult, 1, has_bias=False, target_device=config.device_target)
        self.gmp_fc = Dense(ndf * mult, 1, has_bias=False, target_device=config.device_target)

        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1,
                                 stride=1, has_bias=True, bias_init='uniform')
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.conv = nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1,
                              padding=0, has_bias=False, pad_mode="pad")
        self.model = nn.SequentialCell(model)

        # utils
        self.ada_avg_pool2d = ops.ReduceMean(keep_dims=True)
        self.ada_max_pool2d = ops.ReduceMax(keep_dims=True)
        self.expand_dims = ops.ExpandDims()
        self.concat = ops.Concat(1)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self, img):
        """construct"""
        x = self.model(img)

        gap = self.ada_avg_pool2d(x, (2, 3))
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = self.gap_fc.dense.weight
        gap_weight = self.expand_dims(gap_weight, 2)
        gap_weight = self.expand_dims(gap_weight, 3)
        gap = x * gap_weight

        gmp = self.ada_max_pool2d(x, (2, 3))
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = self.gmp_fc.dense.weight
        gmp_weight = self.expand_dims(gmp_weight, 2)
        gmp_weight = self.expand_dims(gmp_weight, 3)
        gmp = x * gmp_weight

        cam_logit = self.concat((gap_logit, gmp_logit))
        x = self.concat((gap, gmp))
        x = self.leaky_relu(self.conv1x1(x))
        heatmap = self.reduce_sum(x, 1)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap

class DWithLossCell(nn.Cell):
    """Discriminator with loss"""
    def __init__(self, _disGA, _disLA, _disGB, _disLB, _weight):
        super(DWithLossCell, self).__init__(auto_prefix=True)
        self.disGA = _disGA
        self.disGB = _disGB
        self.disLA = _disLA
        self.disLB = _disLB
        self.MSELoss = nn.MSELoss()
        self.weight = _weight[0]

        # utils
        self.ones_like = ops.OnesLike()
        self.zeros_like = ops.ZerosLike()

    def construct(self, real_A, real_B, fake_A2B, fake_B2A):
        """construct"""
        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        D_ad_loss_GA = self.MSELoss(real_GA_logit, self.ones_like(real_GA_logit)) \
                       + self.MSELoss(fake_GA_logit, self.zeros_like(fake_GA_logit))
        D_ad_cam_loss_GA = self.MSELoss(real_GA_cam_logit, self.ones_like(real_GA_cam_logit)) \
                           + self.MSELoss(fake_GA_cam_logit, self.zeros_like(fake_GA_cam_logit))
        D_ad_loss_LA = self.MSELoss(real_LA_logit, self.ones_like(real_LA_logit)) \
                       + self.MSELoss(fake_LA_logit, self.zeros_like(fake_LA_logit))
        D_ad_cam_loss_LA = self.MSELoss(real_LA_cam_logit, self.ones_like(real_LA_cam_logit)) \
                           + self.MSELoss(fake_LA_cam_logit, self.zeros_like(fake_LA_cam_logit))
        D_ad_loss_GB = self.MSELoss(real_GB_logit, self.ones_like(real_GB_logit)) \
                       + self.MSELoss(fake_GB_logit, self.zeros_like(fake_GB_logit))
        D_ad_cam_loss_GB = self.MSELoss(real_GB_cam_logit, self.ones_like(real_GB_cam_logit)) \
                           + self.MSELoss(fake_GB_cam_logit, self.zeros_like(fake_GB_cam_logit))
        D_ad_loss_LB = self.MSELoss(real_LB_logit, self.ones_like(real_LB_logit)) \
                       + self.MSELoss(fake_LB_logit, self.zeros_like(fake_LB_logit))
        D_ad_cam_loss_LB = self.MSELoss(real_LB_cam_logit, self.ones_like(real_LB_cam_logit)) \
                           + self.MSELoss(fake_LB_cam_logit, self.zeros_like(fake_LB_cam_logit))

        D_loss_A = self.weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        Discriminator_loss = D_loss_A + D_loss_B
        return Discriminator_loss

class GWithLossCell(nn.Cell):
    """generator with loss"""
    def __init__(self, generator, _disGA, _disLA, _disGB, _disLB, weights):
        super(GWithLossCell, self).__init__(auto_prefix=True)
        self.generator = generator
        self.disGA = _disGA
        self.disGB = _disGB
        self.disLA = _disLA
        self.disLB = _disLB
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.adv_weight = weights[0]
        self.cycle_weight = weights[1]
        self.identity_weight = weights[2]
        self.cam_weight = weights[3]

        self.oneslike = ops.OnesLike()
        self.zeroslike = ops.ZerosLike()

    def construct(self, real_A, real_B):
        """construct"""
        A2B, A2B_cam, B2A, B2A_cam, A2B2A, B2A2B, A2A, A2A_cam, B2B, B2B_cam = self.generator(real_A, real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, self.oneslike(fake_GA_logit))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, self.oneslike(fake_GA_cam_logit))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, self.oneslike(fake_LA_logit))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, self.oneslike(fake_LA_cam_logit))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, self.oneslike(fake_GB_logit))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, self.oneslike(fake_GB_cam_logit))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, self.oneslike(fake_LB_logit))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, self.oneslike(fake_LB_cam_logit))

        G_recon_loss_A = self.L1_loss(A2B2A, real_A)
        G_recon_loss_B = self.L1_loss(B2A2B, real_B)

        G_identity_loss_A = self.L1_loss(A2A, real_A)
        G_identity_loss_B = self.L1_loss(B2B, real_B)

        G_cam_loss_A = self.BCE_loss(B2A_cam, self.oneslike(B2A_cam)) \
                       + self.BCE_loss(A2A_cam, self.zeroslike(A2A_cam))
        G_cam_loss_B = self.BCE_loss(A2B_cam, self.oneslike(A2B_cam)) \
                       + self.BCE_loss(B2B_cam, self.zeroslike(B2B_cam))

        G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) \
                   + self.cycle_weight * G_recon_loss_A \
                   + self.identity_weight * G_identity_loss_A \
                   + self.cam_weight * G_cam_loss_A
        G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) \
                   + self.cycle_weight * G_recon_loss_B \
                   + self.identity_weight * G_identity_loss_B \
                   + self.cam_weight * G_cam_loss_B

        Generator_loss = G_loss_A + G_loss_B
        return A2B, B2A, Generator_loss
