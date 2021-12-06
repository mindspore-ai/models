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

"""DBPN GAN loss"""

from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F

from src.util.utils import gram_matrix
from src.vgg19.define import vgg19


class WithLossCellPretrainedG(nn.Cell):
    """Loss for pretrain
    use GeneratorLoss to measure the MSELoss
    Args:
        netG: use the generator to rebuild sr image
    Outputs:
        Tensor
    """
    def __init__(self, netG):
        super(WithLossCellPretrainedG, self).__init__()
        self.netG = netG
        self.pixel_criterion = nn.MSELoss()

    def construct(self, hr, lr):
        """pretrained loss
        Args:
            hr(Tensor): high resolution image
        Outputs:
            Tensor
        """
        sr = self.generator(lr)
        psnr_loss = self.pixel_criterion(hr, sr)
        return psnr_loss


class WithLossCellG(nn.Cell):
    """fine-tune Generotor loss
    use GeneratorLoss to measure the MSELoss
    Args:
        netG(Cell): use the generator to rebuild sr image
        netD(Cell): use the discriminator to measure fake_data and
    Outputs:
        Tensor
    """
    def __init__(self, netD, netG):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.adversarial_criterion = nn.BCELoss(reduction="mean")
        self.mse_loss = nn.MSELoss()
        self.vgg = vgg19()
        self.w1 = 0.1
        self.w2 = 0.2
        self.w3 = 0.001
        self.w4 = 1

    def construct(self, hr, lr):
        """compute fine tune loss
        the loss can divide four parts: adversarial loss, content loss,
        style loss, perceptual loss
        Args:
           hr(Tensor): high resolution image
           lr(Tensor): low resolution image
        Outputs:
           Tensor
        """
        ones = ops.Ones()
        batch_num = lr.shape[0]
        fake_data = self.netG(lr)
        out1 = self.netD(fake_data)
        label1 = ones(out1.shape, mstype.float32)
        # adversarial loss
        loss1 = self.adversarial_criterion(out1, label1)
        # Content losses
        loss2 = self.mse_loss(fake_data, hr)
        # perceptual loss
        hr_feat = self.vgg(hr)
        sr = F.stop_gradient(fake_data)
        sr_feat = self.vgg(sr)
        vgg_loss = 0
        style_loss = 0
        for i in range(batch_num):
            real = F.stop_gradient(hr_feat[i])
            gram_real = F.stop_gradient(gram_matrix(hr_feat[i]))
            vgg_loss += self.mse_loss(sr_feat[i], real)
            style_loss += self.mse_loss(gram_matrix(sr_feat[i]), gram_real)

        loss = self.w1 * loss2 + self.w3 * loss1 + self.w2 * vgg_loss + self.w4 * style_loss
        return loss


class WithLossCellD(nn.Cell):
    """ WithLossCellD loss
    use GeneratorLoss to measure the MSELoss
    Args:
        netG(Cell): use the generator to rebuild sr image
        netD(Cell): use the discriminator to measure fake_data and
    Outputs:
        Tensor
    """
    def __init__(self, netD, netG):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.loss_fn = nn.BCELoss(reduction='mean')

    def construct(self, hr, lr):
        """compute loss
        Args:
            hr(Tensor): high resolution image
            lr(Tensor): low resolution image
        Outputs:
            Tensor
        """
        ones = ops.Ones()
        zeros = ops.Zeros()

        out1 = self.netD(hr)
        label1 = ones(out1.shape, mstype.float32)
        loss1 = self.loss_fn(out1, label1)

        fake_data = self.netG(lr)
        fake_data = ops.stop_gradient(fake_data)
        out2 = self.netD(fake_data)
        label2 = zeros(out2.shape, mstype.float32)
        loss2 = self.loss_fn(out2, label2)
        return loss1 + loss2
