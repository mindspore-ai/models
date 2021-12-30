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
"""
ClassifierWithLossCell
"""
import mindspore as ms
from mindspore import ops, Parameter, Tensor
from mindspore import nn
from src.model.resnet12 import resnet12


class Classifier(nn.Cell):
    """
    Classifier
    """

    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.encoder = resnet12()
        in_dim = self.encoder.emb_size
        self.classifier = nn.Dense(in_channels=in_dim, out_channels=n_classes, has_bias=False)

    def construct(self, x):
        """
        :param x: data
        :return: logits
        """
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ClassifierWithLossCell(nn.Cell):
    """
    ClassifierWithLossCell
    """

    def __init__(self, net):
        super(ClassifierWithLossCell, self).__init__()
        self.net = net
        self.loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)
        self.acc = Parameter(Tensor(0.0, ms.float32), requires_grad=False)

    def construct(self, img, labels):
        """
        :param img: data
        :param labels: label
        :return: loss cost
        """
        logits = self.net(img)
        ret = ops.Argmax()(logits) == labels
        acc = ops.ReduceMean()(ret.astype(ms.float32))
        self.acc = acc
        return self.loss(logits, labels)
