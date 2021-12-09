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
"""APDrawingGAN WithLossCellD"""

from mindspore import nn, ops
from src.networks.networks_loss import GANLoss


class WithLossCellD(nn.Cell):
    """WithLossCellD"""
    def __init__(self, net_D, net_G, opt, loss_adv=GANLoss(use_lsgan=True), auto_prefix=True):
        super(WithLossCellD, self).__init__(auto_prefix=auto_prefix)
        self.net_D = net_D
        self.net_G = net_G
        self.use_local = opt.use_local
        self.discriminator_local = opt.discriminator_local
        self.criterionGAN = loss_adv
        self.support_non_tensor_inputs = True

        self.addw_eye = opt.addw_eye
        self.addw_nose = opt.addw_nose
        self.addw_mouth = opt.addw_mouth
        self.addw_hair = opt.addw_hair
        self.addw_bg = opt.addw_bg

    def construct(self, *inputs, **kwargs):
        """construct"""
        real_B, _, _, _, _, _, _, real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, real_A_hair, \
        mask, mask2, _, _ = inputs

        ############################### make the correct predict  #############################
        if self.discriminator_local:
            netD, netDLEyel, netDLEyer, netDLNose, netDLMouth, netDLHair, netDLBG \
            = self.net_D(A=real_A, B=real_B, is_fake=False, mask=mask, mask2=mask2)
            loss_D_real = 0
            if self.use_local:
                loss_D_real += self.criterionGAN(netD, True)
                loss_D_real += self.criterionGAN(netDLEyel, True) * self.addw_eye
                loss_D_real += self.criterionGAN(netDLEyer, True) * self.addw_eye
                loss_D_real += self.criterionGAN(netDLNose, True) * self.addw_nose
                loss_D_real += self.criterionGAN(netDLMouth, True) * self.addw_mouth
                loss_D_real += self.criterionGAN(netDLHair, True) * self.addw_hair
                loss_D_real += self.criterionGAN(netDLBG, True) * self.addw_bg
            else:
                loss_D_real = self.criterionGAN(netD, True)
        else:
            netD = self.net_D(A=real_A, B=real_B, is_fake=False, mask=mask, mask2=mask2)
            loss_D_real = self.criterionGAN(netD, True)

        ############################### make the failure/fake predict  #########################
        if self.use_local:
            fake_B, _, _, _, _, _, _ = self.net_G(real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, \
                                                  real_A_mouth, real_A_hair, mask, mask2)
        else:
            fake_B = self.net_G(real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, \
                                real_A_hair, mask, mask2)
        fake_B = ops.stop_gradient(fake_B)

        if self.discriminator_local:
            netD, netDLEyel, netDLEyer, netDLNose, netDLMouth, netDLHair, netDLBG = \
            self.net_D(A=real_A, B=fake_B, is_fake=True, mask=mask, mask2=mask2)
            loss_D_fake = 0
            if self.use_local:
                loss_D_fake += self.criterionGAN(netD, True)
                loss_D_fake += self.criterionGAN(netDLEyel, True) * self.addw_eye
                loss_D_fake += self.criterionGAN(netDLEyer, True) * self.addw_eye
                loss_D_fake += self.criterionGAN(netDLNose, True) * self.addw_nose
                loss_D_fake += self.criterionGAN(netDLMouth, True) * self.addw_mouth
                loss_D_fake += self.criterionGAN(netDLHair, True) * self.addw_hair
                loss_D_fake += self.criterionGAN(netDLBG, True) * self.addw_bg
            else:
                loss_D_fake = self.criterionGAN(netD, True)
        else:
            netD = self.net_D(A=real_A, B=fake_B, is_fake=True, mask=mask, mask2=mask2)
            loss_D_fake = self.criterionGAN(netD, True)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
