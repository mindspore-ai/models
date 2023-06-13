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

import mindspore.numpy as np

from mindspore import Tensor
import mindspore.ops as ops
from mindspore.nn.loss.loss import LossBase
import mindspore.nn as nn


class ImageGradients(nn.Cell):
    def construct(self, images):
        batch_size, depth, height, width = images.shape
        if height == 1:
            dy = ops.fill(images.dtype, (batch_size, depth, 1, width), 0)
        else:
            dy = images[:, :, 1:, :] - images[:, :, : height - 1, :]
            dy_last = ops.fill(images.dtype, (batch_size, depth, 1, width), 0)
            dy = ops.concat((dy, dy_last), 2)

        if width == 1:
            dx = ops.fill(images.dtype, (batch_size, depth, height, 1), 0)
        else:
            dx = images[:, :, :, 1:] - images[:, :, :, : width - 1]
            dx_last = ops.fill(images.dtype, (batch_size, depth, height, 1), 0)
            dx = ops.concat((dx, dx_last), 3)
        return dy, dx


class CustomLoss(LossBase):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.power = ops.Pow()
        self.sum = ops.ReduceSum()
        self.get_grad = ImageGradients()

    def construct(self, output, target):
        n = target.shape[-1] * target.shape[-2]
        target = np.clip(target, 0.1, 10)
        target = np.log(target)

        di = target - output
        di2 = self.power(di, 2)
        l1_loss = self.sum(np.abs(di), (1, 2, 3)) / n

        dy_pred_depth, dx_pred_depth = self.get_grad(output)
        dy_gt_depth, dx_gt_depth = self.get_grad(target)
        edge_loss = (
            self.sum(np.abs(dy_pred_depth - dy_gt_depth), (1, 2, 3))
            + self.sum(np.abs(dx_pred_depth - dx_gt_depth), (1, 2, 3))
        ) / n

        first_term = self.sum(di2, (1, 2, 3)) / n
        second_term = 0.5 * self.power(self.sum(di, (1, 2, 3)), 2) / (n**2)
        loss = l1_loss + edge_loss + first_term - second_term

        return loss.mean()


class CustomLossOriginal(LossBase):
    def __init__(self):
        super().__init__()
        self.power = ops.Pow()
        self.sum = ops.ReduceSum()
        self.get_grad = ImageGradients()

    def construct(self, output, target):
        n = target.shape[-1] * target.shape[-2]
        target = np.clip(target, 0.1, 10)
        target = np.log(target)

        di = target - output
        di2 = self.power(di, 2)

        first_term = self.sum(di2, (1, 2, 3)) / n
        second_term = 0.5 * self.power(self.sum(di, (1, 2, 3)), 2) / (n**2)
        loss = first_term - second_term

        return loss.mean()


def training_loss(pred_depth, ground_truth):
    power = ops.Pow()
    _sum = ops.ReduceSum()
    ground_truth = np.clip(ground_truth, 0.1, 10.0)
    di = np.log(ground_truth) - pred_depth
    di2 = power(di, 2)

    n = pred_depth.shape[0] * pred_depth.shape[1]
    first_term = _sum(di2) / n
    second_term = 0.5 * power(_sum(di), 2) / (n**2)
    loss = first_term - second_term
    return loss.mean()


def threshold_percentage_loss(pred_depth, ground_truth, threshold_val):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10.0)

    d1 = pred_depth / ground_truth
    d2 = ground_truth / pred_depth

    max_d1_d2 = np.maximum(d1, d2)
    bit_map = np.where(max_d1_d2 <= threshold_val, 1.0, 0.0)
    delta = Tensor(bit_map).sum() / (bit_map.shape[0] * bit_map.shape[1])

    return delta


def rmse_linear(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10.0)

    diff = pred_depth - ground_truth
    diff2 = np.power(diff, 2)
    mse = np.sum(diff2) / (pred_depth.shape[0] * pred_depth.shape[1])
    rmse = np.sqrt(mse)

    return rmse


def rmse_log(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    diff = pred_depth - np.log(ground_truth)
    diff2 = np.power(diff, 2)
    mse = np.sum(diff2) / (pred_depth.shape[0] * pred_depth.shape[1])
    rmse = np.sqrt(mse)

    return rmse


def rmse_log_inv(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)
    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    alpha = np.sum((np.log(ground_truth) - pred_depth)) / (pred_depth.shape[0] * pred_depth.shape[1])
    D = np.sum(np.power((pred_depth - np.log(ground_truth) + alpha), 2)) / (pred_depth.shape[0] * pred_depth.shape[1])

    return D


def abs_relative_difference(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)

    ground_truth = np.clip(ground_truth, 0.1, 10)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10)

    abs_relative_diff = np.abs(pred_depth - ground_truth) / ground_truth
    abs_relative_diff = np.sum(abs_relative_diff) / (pred_depth.shape[0] * pred_depth.shape[1])

    return abs_relative_diff


def squared_relative_difference(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10)

    square_relative_diff = np.power(np.abs(pred_depth - ground_truth), 2) / ground_truth
    square_relative_diff = np.sum(square_relative_diff) / (pred_depth.shape[0] * pred_depth.shape[1])
    return square_relative_diff
