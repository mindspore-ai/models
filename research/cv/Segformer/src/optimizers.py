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
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.nn import AdamWeightDecay


def get_optimizer(model: nn.Cell, optimizer: str, lr: float, weight_decay: float = 0.01, epochs: int = 200,
                  step_per_epoch: int = 100, warmup_steps: int = 1500):
    if warmup_steps > 0:
        assert epochs * step_per_epoch > warmup_steps, f"total step size should be greater than warmup_steps, " \
                                                       f"current warmup steps:{warmup_steps}, " \
                                                       f"current total steps:{epochs * step_per_epoch}"
        lr_list = np.full((epochs * step_per_epoch,), lr, dtype=np.float32)
        warmup_lr = np.interp(np.arange(0, warmup_steps), [0, warmup_steps], [0.0, lr])
        lr_list[:warmup_steps] = warmup_lr
        lr = ms.Tensor(lr_list, ms.float32)
    conv_params = list(filter(lambda x: 'conv' in x.name, model.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, model.trainable_params()))
    params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': lr},
              {'params': no_conv_params, 'lr': lr},
              {'order_params': model.trainable_params()}]
    if optimizer == 'adamw':
        optimizer = AdamWeightDecay(params, learning_rate=lr, beta1=0.9, beta2=0.999, eps=1e-8,
                                    weight_decay=weight_decay)
        return optimizer
    raise NotImplementedError
