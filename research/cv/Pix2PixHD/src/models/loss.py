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
# ===========================================================================

"""
    Define losses.
"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean, _get_parallel_mode
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from src.utils.config import config
from .generator_model import Vgg19


class VGGLoss(nn.Cell):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for param in self.vgg.get_parameters():
            param.requires_grad = False

    def construct(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss


class GANLoss(nn.Cell):
    """
    Define GANLoss.
    """

    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEloss(reduction="none")
        self.ones = ops.OnesLike()
        self.zeros = ops.ZerosLike()

    def construct(self, input_data, target_is_real):
        ones_or_zeros = self.ones if target_is_real else self.zeros
        if isinstance(input_data[0], list):
            loss = 0
            for input_i in input_data:
                pred = input_i[-1]
                loss += self.loss(pred, ones_or_zeros(pred))
            return loss
        return self.loss(input_data[-1], ones_or_zeros(input_data[-1]))


class DWithLossCell(nn.Cell):
    """
    Define DWithLossCell.
    """

    def __init__(self, backbone):
        super(DWithLossCell, self).__init__(auto_prefix=True)
        self.is_train = True
        self.concat = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.one_hot = ops.OneHot(axis=1)
        self.cast = ops.Cast()
        self.criterionGAN = GANLoss(use_lsgan=not config.no_lsgan)
        self.netD = backbone.netD
        self.netG = backbone.netG
        self.use_features = backbone.use_features
        self.gen_features = backbone.gen_features
        self.no_vgg_loss = backbone.no_vgg_loss
        self.no_ganFeat_loss = backbone.no_ganFeat_loss
        self.load_features = backbone.load_features
        self.n_layers_D = backbone.n_layers_D
        self.num_D = backbone.num_D
        self.lambda_feat = backbone.lambda_feat
        if self.gen_features:
            self.netE = backbone.netE

    def construct(self, input_label, inst_map, real_image, feat_map):
        if self.use_features:
            if not self.load_features:
                feat_map = self.netE(real_image, inst_map)
            input_concat = self.concat((input_label, feat_map))
        else:
            input_concat = input_label

        fake_image = self.netG(input_concat)
        # Fake Detection and loss
        pred_fake = self.discriminate(input_label, fake_image)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        return loss_D_fake * 0.5 + loss_D_real * 0.5

    def discriminate(self, input_label, img):
        input_concat = self.concat((input_label, img))
        return self.netD(input_concat)


class GWithLossCell(nn.Cell):
    """
    Define GWithLossCell.
    """

    def __init__(self, backbone):
        super(GWithLossCell, self).__init__(auto_prefix=True)
        self.is_train = True
        self.concat = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.one_hot = ops.OneHot(axis=1)
        self.cast = ops.Cast()
        self.criterionGAN = GANLoss(use_lsgan=not config.no_lsgan)
        self.criterionFeat = nn.L1Loss()
        self.netD = backbone.netD
        self.netG = backbone.netG
        self.use_features = backbone.use_features
        self.gen_features = backbone.gen_features
        self.no_vgg_loss = backbone.no_vgg_loss
        self.no_ganFeat_loss = backbone.no_ganFeat_loss
        self.load_features = backbone.load_features
        self.n_layers_D = backbone.n_layers_D
        self.num_D = backbone.num_D
        self.lambda_feat = backbone.lambda_feat
        if self.gen_features:
            self.netE = backbone.netE
        if not self.no_vgg_loss:
            self.criterionVGG = VGGLoss()

    def construct(self, input_label, inst_map, real_image, feat_map):
        if self.use_features:
            if not self.load_features:
                feat_map = self.netE(real_image, inst_map)
            input_concat = self.concat((input_label, feat_map))
        else:
            input_concat = input_label

        fake_image = self.netG(input_concat)
        # GAN loss
        pred_fake = self.discriminate(input_label, fake_image)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        pred_real = self.discriminate(input_label, real_image)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.no_ganFeat_loss:
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_fake[i][j], pred_real[i][j])
                        * self.lambda_feat
                    )

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.lambda_feat

        return loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG

    def discriminate(self, input_label, img):
        input_concat = self.concat((input_label, img))
        return self.netD(input_concat)


class TrainOneStepCell(nn.Cell):
    """
    Define TrainOneStepCell.
    """

    def __init__(self, loss_netD, loss_netG, optimizerD, optimizerG, sens=1, auto_prefix=True):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        # loss network
        self.loss_netD = loss_netD
        self.loss_netD.set_grad()
        self.loss_netD.add_flags(defer_inline=True)

        self.loss_netG = loss_netG
        self.loss_netG.set_grad()
        self.loss_netG.add_flags(defer_inline=True)

        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        self.weights_D = optimizerD.parameters
        self.optimizerD = optimizerD

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        # Parallel processing
        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_G = DistributedGradReducer(self.weights_G, mean, degree)
            self.grad_reducer_D = DistributedGradReducer(self.weights_D, mean, degree)

    def set_sens(self, value):
        self.sens = value

    def construct(self, label, inst, image, feat):
        """
        Define TrainOneStepCel
        """
        d_loss = self.loss_netD(label, inst, image, feat)
        g_loss = self.loss_netG(label, inst, image, feat)

        g_sens = ops.Fill()(ops.DType()(g_loss), ops.Shape()(g_loss), self.sens)
        g_grads = self.grad(self.loss_netG, self.weights_G)(label, inst, image, feat, g_sens)
        g_res = ops.depend(g_loss, self.optimizerG(g_grads))

        d_sens = ops.Fill()(ops.DType()(d_loss), ops.Shape()(d_loss), self.sens)
        d_grads = self.grad(self.loss_netD, self.weights_D)(label, inst, image, feat, d_sens)
        d_res = ops.depend(d_loss, self.optimizerD(d_grads))
        return d_res, g_res

    def update_optimizerG(self, optimizerG):
        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_G = DistributedGradReducer(self.weights_G, mean, degree)
