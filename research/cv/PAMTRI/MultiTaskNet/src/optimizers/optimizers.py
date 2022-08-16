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
"""optimizers"""
import mindspore.nn as nn

def init_optim(optim, params, lr, weight_decay, loss_scale):
    """choose optimizer"""
    if optim == 'adam':
        return nn.Adam(params, learning_rate=lr, weight_decay=weight_decay, loss_scale=loss_scale)
    if optim == 'amsgrad':
        return nn.Adam(params, learning_rate=lr, weight_decay=weight_decay, use_nesterov=True, loss_scale=loss_scale)
    if optim == 'sgd':
        return nn.SGD(params, learning_rate=lr, momentum=0.9, weight_decay=weight_decay, loss_scale=loss_scale)
    if optim == 'rmsprop':
        return nn.RMSProp(params, learning_rate=lr, momentum=0.9, weight_decay=weight_decay, loss_scale=loss_scale)

    raise KeyError("Unsupported optimizer: {}".format(optim))
