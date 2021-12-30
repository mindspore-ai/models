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
MetaEval
"""
import mindspore as ms
from mindspore import nn, ops, Tensor


class MetaEval:
    """
    MetaEval
    """

    def __init__(self, method='cos', temp=5.):
        super(MetaEval, self).__init__()
        # self.encoder = resnet12()
        self.method = method
        self.temp = temp
        self.loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')

    def eval(self, x_shot, x_query, labels, encoder):
        """
        :param x_shot:
        :param x_query:
        :param labels:
        :param encoder:
        :return: acc loss
        """
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = encoder(ops.Concat(0)([x_shot, x_query]))
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
        logits = ops.BatchMatMul()(x_query, x_shot.transpose(0, 2, 1))

        logits = logits * self.temp

        ret = ops.Argmax()(logits) == labels.astype(ms.int32)
        acc = ret.astype(ms.float32).mean()
        n_way = logits.shape[-1]
        loss = self.loss(logits.view(-1, n_way),
                         ops.OneHot()(labels.astype(ms.int32).view(-1), logits.shape[-1],
                                      Tensor(1.0, ms.float32), Tensor(0.0, ms.float32)))
        return acc, loss
