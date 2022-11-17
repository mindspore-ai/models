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
"""PVNet"""
import math

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as ops
from mindspore.common.initializer import HeUniform, Uniform
from mindspore.nn.loss.loss import LossBase

from src.resnet import resnet18


class Resnet18_8s(nn.Cell):
    """Resnet18_8s Network"""
    def __init__(self, ver_dim, seg_dim=2, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32, pretrained_path=None):
        """__init__"""
        super(Resnet18_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               output_stride=8,
                               pretrained_path=pretrained_path,
                               remove_avg_pool_layer=True)

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.conv_init = HeUniform(negative_slope=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

        resnet18_8s.fc = nn.SequentialCell([
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 'pad', 1, has_bias=False, weight_init=self.conv_init),
            nn.BatchNorm2d(fcdim),
            nn.ReLU()])
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s = nn.SequentialCell([
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 'pad', 1, has_bias=False, weight_init=self.conv_init),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1)])
        self.up8sto4s = nn.ResizeBilinear()
        # x4s->64
        self.conv4s = nn.SequentialCell([
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 'pad', 1, has_bias=False, weight_init=self.conv_init),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1)])
        self.up4sto2s = nn.ResizeBilinear()
        # x2s->64
        self.conv2s = nn.SequentialCell([
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 'pad', 1, has_bias=False, weight_init=self.conv_init),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1)])
        self.up2storaw = nn.ResizeBilinear()

        self.convraw = nn.SequentialCell([
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 'pad', 1, has_bias=False, weight_init=self.conv_init),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1, weight_init=self.conv_init, has_bias=True,
                      bias_init=Uniform())])

        for _, m in self.cells_and_names():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1

        self.concat = ops.Concat(axis=1)

    def construct(self, x, feature_alignment=False):
        """construct Network"""
        x2s, x4s, x8s, _, _, xfc = self.resnet18_8s(x)
        fm = self.conv8s(self.concat((xfc, x8s)))
        fm = self.up8sto4s(fm, scale_factor=2, align_corners=True)
        fm = self.conv4s(self.concat((fm, x4s)))
        fm = self.up4sto2s(fm, scale_factor=2, align_corners=True)

        fm = self.conv2s(self.concat((fm, x2s)))
        fm = self.up2storaw(fm, scale_factor=2, align_corners=True)

        x = self.convraw(self.concat((fm, x)))

        seg_pred = x[:, :self.seg_dim, :, :]
        ver_pred = x[:, self.seg_dim:, :, :]

        return seg_pred, ver_pred


class NetworkWithLossCell(nn.Cell):
    """NetworkWithLossCell"""
    def __init__(self, net, cls_num=2):
        """__init__"""
        super(NetworkWithLossCell, self).__init__()
        self.net = net
        self.loss_fn = PVNetLoss(cls_num=cls_num)

    def construct(self, image, train_mask, train_vertex, vertex_weight):
        """construct"""
        seg_pred, vertex_pred = self.net(image)

        return self.loss_fn(seg_pred, vertex_pred, train_mask, train_vertex, vertex_weight)


class PVNetLoss(LossBase):
    """PVNetLoss"""
    def __init__(self, cls_num=2, sigma=1.0, normalize=True, reduce=True, beta=1.0):
        """__init__"""
        super(PVNetLoss, self).__init__()
        self.cls_num = cls_num
        self.sigma = sigma
        self.normalize = normalize
        self.reduce = reduce
        self.beta = beta

        self.sum = ops.ReduceSum(keep_dims=True)
        self.abs = ops.Abs()
        self.pow = ops.Pow()
        self.cast = ops.Cast()
        self.reduce_mean = ops.ReduceMean(False)
        self.loss_ce = nn.loss.SoftmaxCrossEntropyWithLogits()
        self.on_value = mindspore.Tensor(1.0, mindspore.float32)
        self.off_value = mindspore.Tensor(0.0, mindspore.float32)
        self.one_hot = ops.OneHot(axis=-1)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.tensor_summary = ops.ScalarSummary()
        self.image_summary = ops.ImageSummary()
        self.argmax = ops.Argmax(axis=1)
        self.equal = ops.Equal()

    def compute_precision_recall(self, n, h, w, scores, target, reduce=True):
        """compute_precision_recall"""
        preds = self.argmax(scores)
        preds = self.cast(preds, mindspore.float32)

        tp = preds * target
        fp = preds * (1 - target)
        fn = (1 - preds) * target

        tp = self.sum(tp.view(n, -1), 1)
        fn = self.sum(fn.view(n, -1), 1)
        fp = self.sum(fp.view(n, -1), 1)

        precision = (tp + 1) / (tp + fp + 1)
        recall = (tp + 1) / (tp + fn + 1)

        if reduce:
            precision, recall = self.reduce_mean(precision), self.reduce_mean(recall)
        return precision, recall

    def construct(self, seg_pred, vertex_pred, train_mask, train_vertex, vertex_weight):
        """construct"""
        # n, _, h, w = vertex_pred.shape
        seg_pred = self.cast(seg_pred, mindspore.float32)
        vertex_pred = self.cast(vertex_pred, mindspore.float32)
        train_mask = self.cast(train_mask, mindspore.int64)
        train_vertex = self.cast(train_vertex, mindspore.float32)
        vertex_weight = self.cast(vertex_weight, mindspore.float32)

        seg_pred = self.transpose(seg_pred, (0, 2, 3, 1))
        seg_pred = self.reshape(seg_pred, (-1, self.cls_num))
        train_mask = self.reshape(train_mask, (-1,))
        one_hot_labels = self.one_hot(train_mask, self.cls_num, self.on_value, self.off_value)
        loss_seg = self.loss_ce(seg_pred, one_hot_labels)  # softmax
        loss_seg = self.reduce_mean(loss_seg)

        loss_vertex = self.smooth_l1_loss(vertex_pred, train_vertex, vertex_weight)
        # precision, recall = self.compute_precision_recall(n, h, w, seg_pred, train_mask)
        return loss_seg + loss_vertex

    def smooth_l1_loss(self, vertex_pred, vertex_targets, vertex_weights):
        """
            :param self:            object instance
            :param vertex_pred:     [b,vn*2,h,w], default vn = 9
            :param vertex_targets:  [b,vn*2,h,w], default vn = 9
            :param vertex_weights:  [b,1,h,w]
        """
        b, ver_dim, _, _ = vertex_pred.shape
        sigma_2 = self.cast(self.sigma ** 2, mindspore.float32)
        vertex_diff = vertex_pred - vertex_targets
        diff = vertex_weights * vertex_diff
        abs_diff = self.abs(diff)
        smooth_sign = (abs_diff < self.beta) / sigma_2
        smooth_sign = self.cast(smooth_sign, mindspore.float32)
        in_loss = self.pow(diff, 2) * (sigma_2 / 2.0) * smooth_sign + (abs_diff - (0.5 / sigma_2)) * (1. - smooth_sign)

        if self.normalize:
            in_loss = self.sum(in_loss.reshape(b, -1), 1) / \
                      (ver_dim * self.sum(vertex_weights.reshape(b, -1), 1) + 1e-3)

        if self.reduce:
            in_loss = self.reduce_mean(in_loss)

        return in_loss
