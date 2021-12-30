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
"""trainingcell.py"""
import os
import psutil
import numpy as np
import mindspore as ms
import mindspore.ops as P

from mindspore import nn
from mindspore import ParameterTuple, Tensor, Parameter


def show_memory_info(hint=""):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")


class CriterionWithNet(nn.Cell):
    """
    class of criterion with network
    """
    def __init__(self, backbone, ce_loss, tri_loss, loss_func='id'):
        super(CriterionWithNet, self).__init__()
        self._backbone = backbone
        self.loss_func = loss_func
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss
        self.wg = Parameter(Tensor(np.array([0.0]), dtype=ms.float32),\
            name="wg", requires_grad=False)

        # self.total_loss = 0.0
        # self.wg = 0.0

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def construct(self, img1, img2, label1, label2, adj, modal=0):
        """
        function of constructing
        """
        out_graph = None

        if self._backbone.nheads > 0:
            feat, _, out, out_att, out_graph = self._backbone(
                img1, x2=img2, adj=adj, modal=modal)
        else:
            feat, _, out, out_att = self._backbone(
                img1, x2=img2, modal=modal)

        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        loss_id = self._ce_loss(out, label_)
        loss_tri = self._tri_loss(feat, label)

        if self.loss_func == 'tri':
            loss_total = loss_tri
        elif self.loss_func == 'id+tri':
            loss_total = loss_id + loss_tri
        else:
            loss_total = loss_id

        if self._backbone.part > 0:
            loss_p = self._ce_loss(out_att, label_)
            loss_total = loss_total + loss_p

        if self._backbone.nheads > 0:
            loss_g = P.NLLLoss("mean")(out_graph, label_,
                                       P.Ones()((out_graph.shape[1]), ms.float32))
            loss_total = loss_total + self.wg * loss_g[0]

        return loss_total

    @property
    def backbone_network(self):
        return self._backbone


class OptimizerWithNetAndCriterion(nn.Cell):
    """
    class of optimization methods
    """
    def __init__(self, network, optimizer):
        super(OptimizerWithNetAndCriterion, self).__init__()
        self.network = network
        self.weights = ParameterTuple(optimizer.parameters)
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)

    def construct(self, x1, x2, y1, y2, adj):
        loss = self.network(x1, x2, y1, y2, adj)
        weights = self.weights
        grads = self.grad(self.network, weights)(x1, x2, y1, y2, adj)
        loss = P.Depend()(loss, self.optimizer(grads))
        return loss
