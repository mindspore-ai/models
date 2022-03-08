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

"""loss"""

import os
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
import src.vgg16 as vgg16
from src.inceptionv3 import inceptionv3


def calc_content_loss(input_x, target):
    """content loss"""
    mse_loss = nn.MSELoss(reduction='mean')
    return mse_loss(input_x, target)


def calc_style_loss(platform, input_x, target):
    """style loss"""
    mse_loss = nn.MSELoss(reduction='mean')
    op_matmul = ops.BatchMatMul(transpose_b=True)
    b, c, h, w = input_x.shape  # channel height width
    if platform == 'Ascend':
        input_x = input_x.astype(mstype.float16)
        target = target.astype(mstype.float16)
    input_x = input_x.reshape(b, c, h * w)
    target = target.reshape(b, c, h * w)
    # Compute gram matrix
    input_x = op_matmul(input_x / h, input_x / w)
    target = op_matmul(target / h, target / w)
    style_loss = mse_loss(input_x, target)
    return style_loss


class StyleTransferLoss(nn.Cell):
    """Loss for Style Transfer Model"""

    def __init__(self, args, transfer_net, ckpt_path):
        self.platform = args.platform
        super(StyleTransferLoss, self).__init__()
        self.content_layers = str.split(args.content_layers, ',')
        self.content_each_weight = [float(i) for i in str.split(args.content_each_weight, ',')]
        self.style_layers = str.split(args.style_layers, ',')
        self.style_each_weight = [float(i) for i in str.split(args.style_each_weight, ',')]
        self.style_weight = args.style_weight
        self.transfer_net = transfer_net
        self.meanshift = MeanShift()
        vgg_ckpt = os.path.join(ckpt_path, 'vgg16.ckpt')
        inception_ckpt = os.path.join(ckpt_path, 'inceptionv3.ckpt')
        self.vgg_conv1 = vgg16.vgg16_conv1(vgg_ckpt)
        self.vgg_conv2 = vgg16.vgg16_conv2(vgg_ckpt)
        self.vgg_conv3 = vgg16.vgg16_conv3(vgg_ckpt)
        self.vgg_conv4 = vgg16.vgg16_conv4(vgg_ckpt)
        self.inception = inceptionv3(inception_ckpt)
        for p in self.meanshift.get_parameters():
            p.requires_grad = False

    def vgg_feat(self, x):
        """vgg feature"""
        conv1 = self.vgg_conv1(x)
        conv2 = self.vgg_conv2(x)
        conv3 = self.vgg_conv3(x)
        conv4 = self.vgg_conv4(x)
        feat = {'vgg_16/conv1': conv1, 'vgg_16/conv2': conv2, 'vgg_16/conv3': conv3, 'vgg_16/conv4': conv4}
        return feat

    def construct(self, content_img, style_img):
        """construct"""
        c_img = (content_img + 1.0) / 2.0
        c_img = self.meanshift(c_img)
        s_img = (style_img + 1.0) / 2.0
        s_img = self.meanshift(s_img)
        s_in_feat = self.inception(s_img)
        stylied_img = self.transfer_net(content_img, s_in_feat)

        sty_img = (stylied_img + 1.0) / 2.0
        sty_img = self.meanshift(sty_img)
        c_feat = self.vgg_feat(c_img)
        s_feat = self.vgg_feat(s_img)
        sty_feat = self.vgg_feat(sty_img)

        content_loss = 0
        style_loss = 0
        for index, item in enumerate(self.content_layers):
            content_loss += self.content_each_weight[index] * calc_content_loss(sty_feat[item], c_feat[item])
        for index, item in enumerate(self.style_layers):
            style_loss += self.style_each_weight[index] * calc_style_loss(self.platform, sty_feat[item], s_feat[item])

        all_loss = content_loss + self.style_weight * style_loss
        return all_loss


class MeanShift(nn.Cell):
    """"Meanshift operation"""

    def __init__(self, rgb_range=1, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), sign=-1):
        super(MeanShift, self).__init__()
        std = Tensor(norm_std, mstype.float32)
        eye = ops.Eye()
        newe = eye(3, 3, mstype.float32).view(3, 3, 1, 1)
        new_std = std.view(3, 1, 1, 1)
        weight = Tensor(newe, mstype.float32) / Tensor(new_std, mstype.float32)
        bias = sign * rgb_range * Tensor(norm_mean, mstype.float32) / std
        self.meanshift = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1,
                                   has_bias=True, weight_init=weight, bias_init=bias)

    def construct(self, x):
        """construct"""
        out = self.meanshift(x)
        return out
