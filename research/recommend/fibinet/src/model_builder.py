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
"""components of model building."""

import os
from src.fibinet import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, FiBiNetModel
from src.callbacks import LossCallBack


def get_fibinet_net(config):
    """
    Get network of fibinet model.
    """
    fibinet_net = FiBiNetModel(config)

    loss_net = NetWithLossClass(fibinet_net, config)
    train_net = TrainStepWrap(loss_net, sparse=config.sparse)
    eval_net = PredictWithSigmoid(fibinet_net)

    return train_net, eval_net


class ModelBuilder():
    """
    ModelBuilder
    """
    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_train_hook(self):
        hooks = []
        callback = LossCallBack()
        hooks.append(callback)

        if int(os.getenv('DEVICE_ID')) == 0:
            pass
        return hooks

    def get_net(self, config):
        return get_fibinet_net(config)
