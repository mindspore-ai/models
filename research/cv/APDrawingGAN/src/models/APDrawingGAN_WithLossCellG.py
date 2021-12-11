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
"""APDrawingGAN WithLossCellG"""

from mindspore import nn, ops
from src.networks.networks_loss import GANLoss
from src.networks.controller import define_G


class WithLossCellG(nn.Cell):
    """WithLossCellG"""
    def __init__(self, net_D, net_G, opt, loss_adv=GANLoss(use_lsgan=True), loss_l1=nn.L1Loss(), auto_prefix=True):
        super(WithLossCellG, self).__init__(auto_prefix=auto_prefix)
        self.net_D = net_D
        self.net_G = net_G
        self.support_non_tensor_inputs = True

        self.criterionGAN = loss_adv
        self.criterionL1 = loss_l1
        self.no_l1_loss = False
        self.isTrain = True
        self.lambda_chamfer = opt.lambda_chamfer
        self.lambda_chamfer2 = opt.lambda_chamfer2
        self.lambda_L1 = opt.lambda_L1
        self.use_local = opt.use_local
        self.discriminator_local = opt.discriminator_local
        self.no_G_local_loss = opt.no_G_local_loss
        self.lambda_local = opt.lambda_local

        self.addw_eye = opt.addw_eye
        self.addw_nose = opt.addw_nose
        self.addw_mouth = opt.addw_mouth
        self.addw_hair = opt.addw_hair
        self.addw_bg = opt.addw_bg

        if self.isTrain:
            self.nc = 1
            self.netDT1 = define_G(self.nc, self.nc, opt.ngf, opt.netG_dt, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain)
            self.netDT2 = define_G(self.nc, self.nc, opt.ngf, opt.netG_dt, opt.norm,
                                   not opt.no_dropout, opt.init_type, opt.init_gain)

            self.netLine1 = define_G(self.nc, self.nc, opt.ngf, opt.netG_line, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain)
            self.netLine2 = define_G(self.nc, self.nc, opt.ngf, opt.netG_line, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain)

    def _getaddw(self, local_name):
        """
        get additional weight for each region
        """
        addw = 1
        if local_name in ['DLEyel', 'DLEyer', 'eyel', 'eyer']:
            addw = self.addw_eye
        elif local_name in ['DLNose', 'nose']:
            addw = self.addw_nose
        elif local_name in ['DLMouth', 'mouth']:
            addw = self.addw_mouth
        elif local_name in ['DLHair', 'hair']:
            addw = self.addw_hair
        elif local_name in ['DLBG', 'bg']:
            addw = self.addw_bg
        return addw

    def _cal_loss_chamfer_all(self, real_B, fake_B, dt1gt, dt2gt):
        """
        calculate the distance transform loss
        """
        if fake_B.shape[1] == 3:
            tmp = fake_B[:, 0, ...] * 0.299 + fake_B[:, 1, ...] * 0.587 + fake_B[:, 2, ...] * 0.114
            fake_B_gray = tmp.unsqueeze(1)
        else:
            fake_B_gray = fake_B

        if real_B.shape[1] == 3:
            tmp = real_B[:, 0, ...] * 0.299 + real_B[:, 1, ...] * 0.587 + real_B[:, 2, ...] * 0.114
            real_B_gray = tmp.unsqueeze(1)
        else:
            real_B_gray = real_B
        bs = real_B_gray.shape[0]
        loss_chamfer1 = self._cal_loss_chamfer1(bs, real_B_gray, fake_B_gray)
        loss_chamfer2 = self._cal_loss_chamfer2(bs, fake_B_gray, dt1gt, dt2gt)
        return loss_chamfer1 + loss_chamfer2

    def _cal_loss_chamfer1(self, bs, real_B_gray, fake_B_gray):
        """
        calculate the distance transform loss1
        """
        real_B_gray_line1 = self.netLine1(real_B_gray)
        real_B_gray_line2 = self.netLine2(real_B_gray)

        dt1 = self.netDT1(fake_B_gray)
        dt2 = self.netDT2(fake_B_gray)
        dt1 = dt1 / 2.0 + 0.5
        dt2 = dt2 / 2.0 + 0.5

        logical_and = ops.LogicalAnd()

        temp1 = (real_B_gray < 0)
        temp2 = (real_B_gray_line1 < 0)
        temp3 = (real_B_gray >= 0)
        temp4 = (real_B_gray_line2 >= 0)

        a1 = logical_and(temp1, temp2)
        a2 = logical_and(temp3, temp4)
        dttemp1 = dt1*a1
        dttemp2 = dt2*a2

        sum1 = dttemp1.sum()
        sum2 = dttemp2.sum()

        totoal = sum1 + sum2
        loss_G_chamfer1 = (totoal / bs) * self.lambda_chamfer
        return loss_G_chamfer1

    def _cal_loss_chamfer2(self, bs, fake_B_gray, dt1gt, dt2gt):
        """
        calculate the distance transform loss2
        """
        fake_B_gray_line1 = self.netLine1(fake_B_gray)
        fake_B_gray_line2 = self.netLine2(fake_B_gray)

        logical_and = ops.LogicalAnd()

        temp1 = (fake_B_gray < 0)
        temp2 = (fake_B_gray_line1 < 0)
        temp3 = (fake_B_gray >= 0)
        temp4 = (fake_B_gray_line2 >= 0)

        a1 = logical_and(temp1, temp2)
        a2 = logical_and(temp3, temp4)

        dttemp1 = dt1gt*a1
        dttemp2 = dt2gt*a2

        sum1 = dttemp1.sum()
        sum2 = dttemp2.sum()

        totoal = sum1 + sum2
        loss_G_chamfer2 = (totoal / bs) * self.lambda_chamfer2
        return loss_G_chamfer2

    def construct(self, *inputs, **kwargs):
        """construct"""
        real_B, real_B_bg, real_B_eyel, real_B_eyer, real_B_nose, real_B_mouth, real_B_hair, \
        real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, real_A_hair, \
        mask, mask2, dt1gt, dt2gt = inputs

        loss_G_local = 0.0
        if self.use_local:
            fake_B, fake_B_eyel, fake_B_eyer, fake_B_nose, fake_B_mouth, fake_B_hair, fake_B_bg = \
            self.net_G(real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, real_A_hair, mask, mask2)

            # calculate the local image l1 loss
            if not self.no_G_local_loss:
                addw = self._getaddw('eyel')
                loss_G_local += self.criterionL1(fake_B_eyel, real_B_eyel) * self.lambda_local * addw
                addw = self._getaddw('eyer')
                loss_G_local += self.criterionL1(fake_B_eyer, real_B_eyer) * self.lambda_local * addw
                addw = self._getaddw('nose')
                loss_G_local += self.criterionL1(fake_B_nose, real_B_nose) * self.lambda_local * addw
                addw = self._getaddw('mouth')
                loss_G_local += self.criterionL1(fake_B_mouth, real_B_mouth) * self.lambda_local * addw
                addw = self._getaddw('hair')
                loss_G_local += self.criterionL1(fake_B_hair, real_B_hair) * self.lambda_local * addw
                addw = self._getaddw('bg')
                loss_G_local += self.criterionL1(fake_B_bg, real_B_bg) * self.lambda_local * addw

        else:
            fake_B = self.net_G(real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth,
                                real_A_hair, mask, mask2)

        # predict the output of Discriminator
        if self.discriminator_local:
            netD, netDLEyel, netDLEyer, netDLNose, netDLMouth, netDLHair, netDLBG = \
            self.net_D(A=real_A, B=fake_B, is_fake=False, mask=mask, mask2=mask2)
            loss_G_GAN = 0
            # calculate the adversarial loss
            if self.use_local:
                loss_G_GAN += self.criterionGAN(netD, True)
                loss_G_GAN += self.criterionGAN(netDLEyel, True) * self.addw_eye
                loss_G_GAN += self.criterionGAN(netDLEyer, True) * self.addw_eye
                loss_G_GAN += self.criterionGAN(netDLNose, True) * self.addw_nose
                loss_G_GAN += self.criterionGAN(netDLMouth, True) * self.addw_mouth
                loss_G_GAN += self.criterionGAN(netDLHair, True) * self.addw_hair
                loss_G_GAN += self.criterionGAN(netDLBG, True) * self.addw_bg
            else:
                loss_G_GAN = self.criterionGAN(netD, True)
        else:
            netD = self.net_D(A=real_A, B=fake_B, is_fake=False, mask=mask, mask2=mask2)
            loss_G_GAN = self.criterionGAN(netD, True)

        # calculate the global image l1 loss
        loss_G_L1 = 0.0
        if not self.no_l1_loss:
            loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1

        # calculate the distance transform loss
        loss_chamfer = 0.0
        if self.isTrain:
            loss_chamfer = self._cal_loss_chamfer_all(real_B, fake_B, dt1gt, dt2gt)

        loss_G = loss_G_GAN + loss_G_L1 + loss_G_local + loss_chamfer

        return loss_G
