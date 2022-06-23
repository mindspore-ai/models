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
"""define network with loss function"""

from mindspore.ops import operations as ops
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.nn as nn
from mindspore import dtype as mstype, context
from src.network.util import ReconstructionLoss

class Rounding(nn.Cell):
    """the rounding operation"""
    def __init__(self):
        super(Rounding, self).__init__()
        self.round = ops.Round()

    def construct(self, x):
        x = C.clip_by_value(x, 0, 1)
        x = self.round(x * 255.) / 255.
        return x


class Quantization(nn.Cell):
    """the quantization operation"""
    def __init__(self):
        super(Quantization, self).__init__()
        self.rounding = Rounding()

    def construct(self, x):
        x1 = F.stop_gradient(x)
        rounded = self.rounding(x1)
        residual = rounded - x1
        return x + residual

class IRN_loss(nn.Cell):
    """the irn network with redefined loss function"""

    def __init__(self, net_G, opt):
        super(IRN_loss, self).__init__()

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32

        self.cast = ops.Cast()
        self.netG = net_G

        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.img_visual = {}

        self.is_train = opt['is_train']
        if self.is_train:
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

        self.stdnormal = ops.StandardNormal(10)
        self.ms_sum = ops.ReduceSum()
        self.cat = ops.Concat(1)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.Quantization = Quantization()

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = self.cast(self.reshape(z, (out.shape[0], -1)), self.cast_type)
        l_forw_ce = self.train_opt['lambda_ce_forw'] * self.ms_sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_ce

    def gaussian_batch(self, dims):
        return self.cast(self.stdnormal(dims), self.cast_type)

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)
        return l_back_rec

    def backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        return x_samples_image[0]

    def test(self, ref_L, real_H):
        """model testing"""
        output = self.netG(real_H)
        Lshape = ref_L.shape
        zshape = (Lshape[0], Lshape[1] * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3])

        LR = self.Quantization(output[:, :3, :, :])

        gaussian_scale = 1
        T = gaussian_scale * self.gaussian_batch(zshape)
        y_forw = self.cat((LR, T))

        fake_H = self.netG(x=y_forw, rev=True)
        fake_H_image = fake_H[:, :3, :, :]

        return real_H[0], ref_L[0], LR[0], fake_H_image[0]


    def construct(self, ref_L, real_H):
        """construct method"""
        ### forward downscaling
        output = self.netG(x=real_H)
        LR_ref = ref_L

        ##     l_forw_fit  --LR guidance loss
        ##     l_forw_ce   --distribution matching loss
        l_forw_fit, l_forw_ce = self.loss_forward(output[:, :3, :, :], LR_ref, output[:, 3:, :, :])

        ## backward upscaling
        zshape = output[:, 3:, :, :].shape
        LR = self.Quantization(output[:, :3, :, :])

        gaussian_scale = 1
        T = gaussian_scale * self.gaussian_batch(zshape)

        y_ = self.cat((LR, T))

        l_back_rec = self.loss_backward(real_H, y_)

        ## total loss
        loss = l_forw_fit + l_forw_ce + l_back_rec
        return loss
