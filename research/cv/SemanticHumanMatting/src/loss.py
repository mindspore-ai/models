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

"""loss"""
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.nn import LossBase
from mindspore.ops import constexpr
from mindspore.ops import operations as P


@constexpr
def toTensor(x):
    return Tensor(x, dtype=mindspore.float32)


class SoftmaxCrossEntropyLossFromPytorch(nn.Cell):
    """Softmax Cross Entropy Loss"""

    def __init__(self):
        super(SoftmaxCrossEntropyLossFromPytorch, self).__init__()

        self.log_softmax = nn.LogSoftmax(axis=1)
        self.onehot = ops.OneHot()
        self.transpose = ops.Transpose()
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.squeeze = ops.Squeeze(1)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self, x, label):
        x = -self.log_softmax(x)
        label = self.squeeze(label)
        label = self.onehot(label, 3, self.on_value, self.off_value)
        label = self.transpose(label, (0, 3, 1, 2))

        output = x * label
        output = output.sum(axis=1)
        output = output.mean()
        return output


class LossTNet(LossBase):
    """The loss of T-Net phase"""

    def __init__(self, reduction="mean"):
        super(LossTNet, self).__init__(reduction)

        self.criterion = SoftmaxCrossEntropyLossFromPytorch()

    def construct(self, trimap_pre, trimap_gt):
        l_t = self.criterion(trimap_pre, trimap_gt)
        return l_t


class LossMNet(LossBase):
    """The loss of M-Net phase"""

    def __init__(self, reduction="mean"):
        super(LossMNet, self).__init__(reduction)

        self.criterion = SoftmaxCrossEntropyLossFromPytorch()
        self.sqrt = P.Sqrt()
        self.pow = P.Pow()
        self.cat = P.Concat(axis=1)
        self.add = P.Add()

    def construct(self, img, alpha_pre, alpha_gt):
        eps = 1e-6
        l_alpha = self.sqrt(self.pow(alpha_pre - alpha_gt, 2.0) + eps).mean()

        fg = self.cat((alpha_gt, alpha_gt, alpha_gt)) * img
        fg_pre = self.cat((alpha_pre, alpha_pre, alpha_pre)) * img
        l_composition = self.sqrt(self.add(self.pow(fg - fg_pre, 2.0), eps)).mean()

        l_p = toTensor(0.5) * l_alpha + toTensor(0.5) * l_composition

        return l_p


class LossNet(LossBase):
    """The loss of End-to-End phase"""

    def __init__(self, reduction="mean"):
        super(LossNet, self).__init__(reduction)

        self._loss_t_net = LossTNet()
        self._loss_m_net = LossMNet()

    def construct(self, img, trimap_pre, trimap_gt, alpha_pre, alpha_gt):
        l_t = self._loss_t_net(trimap_pre, trimap_gt)
        l_p = self._loss_m_net(img, alpha_pre, alpha_gt)
        loss = l_p + toTensor(0.01) * l_t
        return loss
