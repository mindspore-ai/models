# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import nn, ops, ms_function


class ModelWithLoss(nn.Cell):
    def __init__(self, model, criterion):
        super(ModelWithLoss).__init__()
        self.model = model
        self.criterion = criterion

    @ms_function
    def construct(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model(*data)
        return self.criterion(output_data[1], rewards)

    @ms_function
    def backup(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model.backup(*data)
        return self.criterion(output_data, rewards)


class ModelWithOptimizer(nn.Cell):
    def __init__(self, model_with_loss, optimizer):
        super(ModelWithOptimizer).__init__()
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(self.model_with_loss.backup, None, optimizer.parameters, has_aux=False)

    @ms_function
    def construct(self, *data):
        loss, grads = self.grad_fn(*data)
        grads = ops.clip_by_global_norm(grads, 20)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
