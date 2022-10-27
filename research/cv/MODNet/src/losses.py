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
import math
import numpy as np
import scipy
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops


class MatteLoss(nn.Cell):
    '''Loss for fusion branch'''
    def __init__(self, reduction='mean'):
        super(MatteLoss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def construct(self, pred_matte, gt_matte, image, boundaries, trimap):
        pred_boundary_matte = ops.select(boundaries, trimap, pred_matte)

        matte_l1_loss = self.loss_fn(pred_matte, gt_matte) + 4.0 * self.loss_fn(pred_boundary_matte, gt_matte)
        matte_compositional_loss = self.loss_fn(image * pred_matte, image * gt_matte) \
                                   + 4.0 * self.loss_fn(image * pred_boundary_matte, image * gt_matte)
        matte_loss = 0.5 * matte_l1_loss + 0.5 * matte_compositional_loss
        return matte_loss


def semantic_loss_fn():
    '''Loss for low-resolution branch'''
    return nn.MSELoss(reduction='mean')


def detail_loss_fn():
    '''Loss for high-resolution branch'''
    return nn.L1Loss(reduction='mean')


def matte_loss_fn():
    '''Loss for fusion branch'''
    return MatteLoss(reduction='mean')

class BlurredCell(nn.Cell):
    '''cell for gaussian blur'''

    def __init__(self, scales=1/16, channels=3, kernel_size=3):
        super(BlurredCell, self).__init__()
        self.scales = scales
        self.channels = channels
        self.kernel_size = kernel_size
        padding_size = math.floor(self.kernel_size / 2)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (padding_size, padding_size), (padding_size, padding_size)),
                          mode="REFLECT")
        weight = self._init_kernrl()

        self.blur = nn.Conv2d(channels, channels, kernel_size, group=channels, pad_mode='valid', weight_init=weight)
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, gt_matte):
        gt_matte = ops.interpolate(gt_matte, scales=(1., 1., self.scales, self.scales),
                                   coordinate_transformation_mode="align_corners", mode='bilinear')
        # blur
        gt_matte = self.pad(gt_matte)
        gt_matte = self.blur(gt_matte)

        return gt_matte

    def _init_kernrl(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)
        kernel = kernel[np.newaxis, :]
        kernel = np.stack([kernel]*self.channels)
        return Tensor(kernel, mindspore.float32)

class MODNetLossCell(nn.Cell):
    '''final loss of modnet'''

    def __init__(self,
                 net,
                 semantic_scale=10.0,
                 detail_scale=10.0,
                 matte_scale=1.0
                 ):
        super(MODNetLossCell, self).__init__()
        self.net = net
        self.semantic_loss_fn = semantic_loss_fn()
        self.detail_loss_fn = detail_loss_fn()
        self.matte_loss_fn = matte_loss_fn()
        self.blurer = BlurredCell(scales=1/16, channels=1, kernel_size=3)
        self.semantic_scale = semantic_scale
        self.detail_scale = detail_scale
        self.matte_scale = matte_scale

    def construct(self, image, trimap, gt_matte):

        # forward the model
        pred_semantic, pred_detail, pred_matte = self.net(image, inference=False)

        # calculate the boundary mask from the trimap
        boundaries = ops.logical_or(trimap < 0.5, trimap > 0.5)

        # calculate the semantic loss
        gt_semantic = self.blurer(gt_matte)
        semantic_loss = self.semantic_loss_fn(pred_semantic, gt_semantic)

        # calculate the detail loss
        pred_boundary_detail = ops.select(boundaries, trimap, pred_detail)
        gt_detail = ops.select(boundaries, trimap, gt_matte)
        detail_loss = self.detail_loss_fn(pred_boundary_detail, gt_detail)

        # calculate the matte loss
        matte_loss = self.matte_loss_fn(pred_matte, gt_matte, image, boundaries, trimap)

        # calculate the final loss
        loss = self.semantic_scale * semantic_loss + self.detail_scale * detail_loss + self.matte_scale * matte_loss

        return loss
