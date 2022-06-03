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
"""losses"""

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops


def gram_matrix(feat):
    """gram matrix"""
    b, ch, h, w = feat.shape
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(0, 2, 1)
    gram = ops.BatchMatMul()(feat, feat_t) / (ch * h * w)

    return gram


class WithLossCell(nn.Cell):
    """Wrap the network with loss function to return generator loss"""
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, *inputs):
        """Construct"""
        output = self.network(*inputs)
        return output[0]


class GWithLossCell(nn.Cell):
    """Generator with loss cell"""
    def __init__(self, generator, discriminator, vgg_feat_extractor, cfg):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_feat_extractor = vgg_feat_extractor

        self.l1 = nn.L1Loss()
        self.criterion = nn.BCELoss(reduction='mean')
        self.real_target = Tensor(1.0, mstype.float32)

        self.hole_loss_w = cfg.hole_loss_w
        self.valid_loss_w = cfg.valid_loss_w
        self.perceptual_loss_w = cfg.perceptual_loss_w
        self.style_loss_w = cfg.style_loss_w
        self.adversarial_loss_w = cfg.adversarial_loss_w
        self.intermediate_loss_w = cfg.intermediate_loss_w

    def construct(self, *inputs):
        """construct"""
        ground_truth, mask, edge, gray_image = inputs
        input_image = ground_truth * mask
        input_edge = edge * mask
        input_gray_image = gray_image * mask
        output, projected_image, projected_edge = self.generator(
            input_image,
            ops.Concat(axis=1)((input_edge, input_gray_image)),
            mask
        )

        output_pred, output_edge = self.discriminator(output, gray_image, edge, is_real=False)

        loss_hole = self.l1((1 - mask) * output, (1 - mask) * ground_truth)

        loss_valid = self.l1(mask * output, mask * ground_truth)

        comp = ground_truth * mask + output * (1 - mask)
        vgg_comp = self.vgg_feat_extractor(comp)
        vgg_output = self.vgg_feat_extractor(output)
        vgg_ground_truth = self.vgg_feat_extractor(ground_truth)

        loss_perceptual = 0.0
        for i in range(3):
            loss_perceptual += self.l1(vgg_output[i], vgg_ground_truth[i])
            loss_perceptual += self.l1(vgg_comp[i], vgg_ground_truth[i])

        loss_style = 0.0
        for i in range(3):
            loss_style += self.l1(gram_matrix(vgg_output[i]), gram_matrix(vgg_ground_truth[i]))
            loss_style += self.l1(gram_matrix(vgg_comp[i]), gram_matrix(vgg_ground_truth[i]))

        real_target = self.real_target.expand_as(output_pred)
        loss_adversarial = self.criterion(output_pred, real_target) + self.criterion(output_edge, edge)

        loss_intermediate = self.criterion(projected_edge, edge) + self.l1(projected_image, ground_truth)

        loss_g = (loss_hole.mean() * self.hole_loss_w + loss_valid.mean() * self.valid_loss_w +
                  loss_perceptual.mean() * self.perceptual_loss_w +
                  loss_style.mean() * self.style_loss_w +
                  loss_adversarial.mean() * self.adversarial_loss_w +
                  loss_intermediate.mean() * self.intermediate_loss_w)
        return loss_g, output


class DWithLossCell(nn.Cell):
    """Discriminator with loss cell"""
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.criterion = nn.BCELoss(reduction='mean')
        self.real_target = Tensor(1.0, mstype.float32)
        self.fake_target = Tensor(0.0, mstype.float32)

    def construct(self, *inputs):
        """construct"""
        ground_truth, gray_image, edge, output = inputs
        real_pred, real_pred_edge = self.discriminator(ground_truth, gray_image, edge, is_real=True)
        fake_pred, fake_pred_edge = self.discriminator(output, gray_image, edge, is_real=False)

        real_target = self.real_target.expand_as(real_pred)
        fake_target = self.fake_target.expand_as(fake_pred)

        loss_adversarial = (self.criterion(real_pred, real_target) +
                            self.criterion(fake_pred, fake_target) +
                            self.criterion(real_pred_edge, edge) +
                            self.criterion(fake_pred_edge, edge))
        return loss_adversarial
