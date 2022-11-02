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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" Loss Computation of SPADE """

from mindspore import nn, ops
from src.models.vgg import Vgg19

class GANLoss(nn.Cell):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))
        self.mean = ops.ReduceMean(keep_dims=False)
        self.min = ops.Minimum()

    def construct(self, input_list, target_is_real, for_discriminator=True):
        loss = 0
        for pred_i in input_list:
            if isinstance(pred_i, list):
                pred_i = pred_i[-1]
            if for_discriminator:
                if target_is_real:
                    minval = self.min(pred_i - 1, 0.0)
                    loss_tensor = -self.mean(minval)
                else:
                    minval = self.min(-pred_i - 1, 0.0)
                    loss_tensor = -self.mean(minval)
            else:
                loss_tensor = -self.mean(pred_i)
            bs = 1 if loss_tensor.shape == () else loss_tensor.shape[0]
            new_loss = self.mean(loss_tensor.view(bs, -1), 1)
            loss += new_loss
        return loss / len(input_list)


class VGGLoss(nn.Cell):
    def __init__(self, opt):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(opt)
        for param in self.vgg.get_parameters():
            param.requires_grad = False
        self.vgg.set_train(False)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def construct(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], ops.stop_gradient(y_vgg[i]))
        return loss
