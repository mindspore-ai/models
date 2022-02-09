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
"""loss net"""
import mindspore.nn as nn
import mindspore.ops as ops


class ActorWithLossCell(nn.Cell):
    """
         Basic Actor Loss Net
         Args:
             actor_net (Cell): backbone
             critic_net (Cell): loss net.
          Returns:
              Tensor, output tensor.
    """
    def __init__(self, actor_net, critic_net):
        super(ActorWithLossCell, self).__init__()
        self._backbone = actor_net
        self._loss_fn = critic_net
        self.reduce_mean = ops.ReduceMean()

    def construct(self, state):
        """construct"""
        action = self._backbone(state)
        q_value = self._loss_fn(state, action)
        actor_loss = - self.reduce_mean(q_value)
        return actor_loss


class CriticWithLossCell(nn.Cell):
    """
         Basic Actor Loss Net
         Args:
             critic_network (Cell): backbone
             loss_func (Cell): loss net.
          Returns:
              Tensor, output tensor.
    """
    def __init__(self, critic_network, loss_func):
        super(CriticWithLossCell, self).__init__()
        self._backbone = critic_network
        self._loss_func = loss_func
        self.reduce_mean = ops.ReduceMean()

    def construct(self, state, action, q_target_value):
        """construct"""
        q_value = self._backbone(state, action)
        td_loss = self._loss_func(q_target_value, q_value)
        td_loss = self.reduce_mean(td_loss)
        return td_loss
