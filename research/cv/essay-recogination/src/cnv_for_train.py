#!/bin/bash
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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, ParameterTuple


class MyWithLossCell(nn.Cell):
    def __init__(self, model, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._model = model
        self._loss_fn = loss_fn

    def construct(self, img, label):
        out = self._model(img)
        preds = mnp.transpose(out, (1, 0, 2))

        labels_indices = Tensor(np.zeros((len(label), 2)), ms.int64)
        label = label.astype(dtype=ms.int32)
        sequence_length = Tensor([4050], dtype=ms.int32)
        return self._loss_fn(preds, labels_indices, label, sequence_length)

    @property
    def backbone_network(self):
        return self._model


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optmizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weight = ParameterTuple(network.trainable_params())
        self.optimizer = optmizer
        self.grad = ops.GradOperation(get_by_list=True)
    def construct(self, data, label):
        weight = self.weight
        loss = self.network(data, label)
        grads = self.grad(self.network, weight)(data, label)
        return loss, self.optimizer(grads)
