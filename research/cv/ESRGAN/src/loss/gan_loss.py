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

""""GAN Loss"""

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from src.vgg19.define import vgg19
import numpy as np


class GeneratorLoss(nn.Cell):
    """Loss for generator"""
    def __init__(self, discriminator, generator, vgg_ckpt, args):
        super(GeneratorLoss, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.l1_loss = nn.L1Loss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        ones = ops.Ones()
        zeros = ops.Zeros()
        self.real_labels = ones((int(16 / args.device_num), 1), mstype.float32)
        self.fake_labels = zeros((int(16 / args.device_num), 1), mstype.float32)
        self.vgg = vgg19(vgg_ckpt)
        mean = Tensor(np.array([0.485, 0.456, 0.406]), mindspore.float32)
        reshape = ops.Reshape()
        self.mean = reshape(mean, (1, 3, 1, 1))
        std = Tensor(np.array([0.229, 0.224, 0.225]), mindspore.float32)
        self.std = reshape(std, (1, 3, 1, 1))

    def construct(self, HR_img, LR_img):
        """gloss"""
        # pixel loss
        hr = HR_img
        lr = LR_img
        sr = self.generator(lr)
        loss_pix = self.l1_loss(sr, hr)

        # adversarialloss
        real_d_pred = self.discriminator(hr)
        fake_g_pred = self.discriminator(sr)
        l_g_real = self.adversarial_criterion(real_d_pred - fake_g_pred.mean(), self.fake_labels)
        l_g_fake = self.adversarial_criterion(fake_g_pred - real_d_pred.mean(), self.real_labels)
        adversarial_loss = (l_g_real + l_g_fake) / 2

        # vggloss
        hr = (hr - self.mean) / self.std
        sr = (sr - self.mean) / self.std
        hr_feat = self.vgg(hr)
        sr_feat = self.vgg(sr)
        percep_loss = self.l1_loss(hr_feat, sr_feat)

        g_loss = 0.01 * loss_pix + 1.0 * percep_loss + 0.005 * adversarial_loss
        return g_loss


class DiscriminatorLoss(nn.Cell):
    """Loss for discriminator"""
    def __init__(self, discriminator, generator, args):
        super(DiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        ones = ops.Ones()
        zeros = ops.Zeros()
        self.real_labels = ones((int(16 / args.device_num), 1), mstype.float32)
        self.fake_labels = zeros((int(16 / args.device_num), 1), mstype.float32)

    def construct(self, HR_img, LR_img):
        """dloss"""
        hr = HR_img
        lr = LR_img
        sr = self.generator(lr)
        # real
        fake_d_pred = self.discriminator(sr)
        real_d_pred = self.discriminator(hr)
        l_d_real = self.adversarial_criterion(real_d_pred - fake_d_pred.mean(), self.real_labels)
        # fake
        l_d_fake = self.adversarial_criterion(fake_d_pred - real_d_pred.mean(), self.fake_labels)
        d_loss = (l_d_real + l_d_fake) / 2

        return d_loss
