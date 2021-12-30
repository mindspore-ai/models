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
MetaBaseline
"""
import mindspore as ms
from mindspore import nn, Parameter, ops, Tensor
from src.model.resnet12 import resnet12


class MetaBaseline(nn.Cell):
    """
    MetaBaseline
    """

    def __init__(self, method='cos', temp=5.0, temp_learnable=True):
        super(MetaBaseline, self).__init__()
        self.encoder = resnet12()
        self.method = method
        if temp_learnable:
            self.temp = Parameter(Tensor([temp], ms.float32), requires_grad=True)
        else:
            self.temp = [temp]

    def construct(self, x_shot, x_query):
        """
        :param x_shot:
        :param x_query:
        :return: logit
        """
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(ops.Concat(0)([x_shot, x_query]))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        ########## cross-class bias ############
        bs = x_shot.shape[0]
        fs = x_shot.shape[-1]
        bias = x_shot.view(bs, -1, fs).mean(1) - x_query.mean(1)
        x_query = x_query + ops.ExpandDims()(bias, 1)

        x_shot = x_shot.mean(axis=-2)
        x_shot = ops.L2Normalize(axis=-1)(x_shot)
        x_query = ops.L2Normalize(axis=-1)(x_query)
        logit = ops.BatchMatMul()(x_query, x_shot.transpose(0, 2, 1))

        return logit * self.temp[0]


class MetaBaselineWithLossCell(nn.Cell):
    """
    MetaBaselineWithLossCell
    """

    def __init__(self, net):
        super(MetaBaselineWithLossCell, self).__init__()
        self.net = net
        self.loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)
        self.acc = Parameter(Tensor(0.0, ms.float32), requires_grad=False)

    def construct(self, x_shot, x_query, labels):
        """
        :param x_shot:
        :param x_query:
        :param labels:
        :return: loss
        """
        logits = self.net(x_shot, x_query)
        ret = ops.Argmax()(logits) == labels
        acc = ret.astype(ms.float32).mean()
        self.acc = acc
        n_way = logits.shape[-1]
        return self.loss(logits.view(-1, n_way), labels.astype(ms.int32).view(-1))
