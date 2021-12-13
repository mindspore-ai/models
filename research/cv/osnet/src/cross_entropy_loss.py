# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Define loss function."""

import mindspore.nn as nn
import mindspore.ops as ops


class CrossEntropyLoss(nn.Cell):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, eps=0.1, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps if label_smooth else 0
        self.logsoftmax = nn.LogSoftmax(axis=1)
        self.zeros = ops.Zeros()
        self.expand_dims = ops.ExpandDims()

    def construct(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        depth = log_probs.shape[1]
        onehot = nn.OneHot(depth=depth, axis=-1)
        targets = onehot(targets)
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        return (-targets * log_probs).mean(0).sum()
