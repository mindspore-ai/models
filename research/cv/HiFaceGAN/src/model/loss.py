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
"""Loss function for a model"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.model.vgg import Vgg19


class VGGForLoss(nn.Cell):
    """VGG network for loss computation"""

    def __init__(self, pretrained_path):
        super().__init__()
        vgg_net = Vgg19()
        vgg_net.set_train(False)
        ms.load_checkpoint(pretrained_path, net=vgg_net)

        self.slice1 = nn.SequentialCell([])
        self.slice2 = nn.SequentialCell([])
        self.slice3 = nn.SequentialCell([])
        self.slice4 = nn.SequentialCell([])
        self.slice5 = nn.SequentialCell([])

        # we are using Vgg19 with BatchNorm
        for i in range(3):
            self.slice1.append(vgg_net.layers[i])
        for i in range(3, 9):
            self.slice2.append(vgg_net.layers[i])
        for i in range(9, 16):
            self.slice3.append(vgg_net.layers[i])
        for i in range(16, 29):
            self.slice4.append(vgg_net.layers[i])
        for i in range(29, 42):
            self.slice5.append(vgg_net.layers[i])

    def construct(self, x):
        """Construct forward graph"""
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5


class GANLoss(nn.Cell):
    """GAN loss"""

    def __init__(self):
        super().__init__()
        self.real_label_tensor = ms.Tensor(1.0, dtype=ms.float32)
        self.fake_label_tensor = ms.Tensor(0.0, dtype=ms.float32)

    def get_target_tensor(self, x, target_is_real):
        """Get target tensor filled with ones or zeros"""
        if target_is_real:
            return self.real_label_tensor.expand_as(x)
        return self.fake_label_tensor.expand_as(x)

    def loss(self, x, target_is_real):
        """Compute loss for two tensors"""
        target_tensor = self.get_target_tensor(x, target_is_real)
        mse = nn.MSELoss()
        return mse(x, target_tensor)

    def construct(self, input_list, target_is_real):
        """Construct forward graph"""
        loss = 0
        for pred_i in input_list:
            if isinstance(pred_i, list):
                pred_i = pred_i[-1]
            loss_tensor = self.loss(pred_i, target_is_real)
            bs = 1 if loss_tensor.ndim == 0 else loss_tensor.shape[0]
            new_loss = loss_tensor.view(bs, -1).mean(axis=1)
            loss += new_loss
        return loss / len(input_list)


class VGGLoss(nn.Cell):
    """Perceptual VGG loss"""

    def __init__(self, pretrained_path):
        super().__init__()
        self.vgg = VGGForLoss(pretrained_path)
        self.criterion = nn.L1Loss()
        self.weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]
        self.n_features = 5

    def construct(self, x, y):
        """Construct forward graph"""
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0
        for i in range(self.n_features):
            loss += self.weights[i] * self.criterion(x_vgg[i], ops.stop_gradient(y_vgg[i]))
        return loss


def divide_pred(pred):
    """Divides the prediction for fake and real images from the combined batch"""
    fake, real = [], []
    for single_pred in pred:
        single_fake, single_real = [], []
        for tens in single_pred:
            di = tens.shape[0] // 2
            single_fake.append(tens[:di])
            single_real.append(tens[di:])
        fake.append(single_fake)
        real.append(single_real)
    return fake, real


def discriminate(discriminator, lq, generated, hq):
    """Computes discriminator prediction"""
    fake_concat = ops.Concat(axis=1)((lq, generated))
    real_concat = ops.Concat(axis=1)((lq, hq))

    # In Batch Normalization, the fake and real images are
    # recommended to be in the same batch to avoid disparate
    # statistics in fake and real images.
    # So both fake and real images are fed to D all at once
    fake_and_real = ops.Concat(axis=0)((fake_concat, real_concat))
    discriminator_out = discriminator(fake_and_real)
    pred_fake, pred_real = divide_pred(discriminator_out)
    return pred_fake, pred_real


class GeneratorLoss(nn.Cell):
    """Generator loss"""

    def __init__(self, generator, discriminator, pretrained_vgg_path, use_vgg_loss, use_gan_feat_loss, lambda_feat,
                 lambda_vgg):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gan_loss = GANLoss()
        self.vgg_loss = VGGLoss(pretrained_vgg_path) if use_vgg_loss else None
        self.gan_feat_loss = nn.L1Loss() if use_gan_feat_loss else None
        self.num_discriminators = discriminator.num_D
        self.lambda_feat = lambda_feat
        self.lambda_vgg = lambda_vgg

    def construct(self, lq, hq):
        """Construct forward graph"""
        generated = self.generator(lq)
        pred_fake, pred_real = discriminate(self.discriminator, lq, generated, hq)
        gan_loss = self.gan_loss(pred_fake, True)

        _, _, h, w = generated.shape
        if self.vgg_loss is None or h < 64 or w < 64:
            vgg_loss = 0
        else:
            vgg_loss = self.vgg_loss(generated, hq) * self.lambda_vgg

        gan_feat_loss = 0
        if self.gan_feat_loss is not None:
            for i in range(self.num_discriminators):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.gan_feat_loss(pred_fake[i][j], ops.stop_gradient(pred_real[i][j]))
                    gan_feat_loss += unweighted_loss * self.lambda_feat / self.num_discriminators

        loss = vgg_loss + gan_loss + gan_feat_loss
        return loss, vgg_loss, gan_loss, gan_feat_loss, generated


class DiscriminatorLoss(nn.Cell):
    """Discriminator loss"""

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.gan_loss = GANLoss()

    def construct(self, lq, hq, generated):
        """Construct forward graph"""
        pred_fake, pred_real = discriminate(self.discriminator, lq, generated, hq)
        loss_fake = self.gan_loss(pred_fake, False)
        loss_real = self.gan_loss(pred_real, True)
        loss = loss_fake + loss_real
        return loss, loss_fake, loss_real
