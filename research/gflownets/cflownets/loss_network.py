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
"""
Loss and train step
"""
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C

from config import CONFIG as cfg


class FlowMatchLoss(nn.Cell):
    """
    flow match loss function net
    """

    def __init__(self, network):
        super(FlowMatchLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.ones = P.Ones()
        self.epi = self.ones((cfg['batch_size'], cfg['max_episode_steps']), mstype.float32)
        self.mse = nn.MSELoss(reduction='none')
        self.log = P.Log()
        self.exp = P.Exp()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.cat_axis1 = P.Concat(axis=1)
        self.cat_axis3 = P.Concat(axis=3)
        self.mean = P.ReduceMean(keep_dims=False)


    def construct(self, inflow_state, inflow_action, outflow_state, outflow_action, not_done, done_true, reward):
        # inflow
        inflow_sa = self.cat_axis3((inflow_state, inflow_action))
        edge_inflow = self.network(inflow_sa).reshape(cfg['batch_size'], cfg['max_episode_steps'], -1)
        inflow = self.log(self.reduce_sum(self.exp(self.log(edge_inflow)), -1) + self.epi)

        # outflow
        outflow_sa = self.cat_axis3((outflow_state, outflow_action))
        edge_outflow = self.network(outflow_sa).reshape(cfg['batch_size'], cfg['max_episode_steps'], -1)
        outflow = self.log(self.reduce_sum(self.exp(self.log(edge_outflow)), -1) + self.epi)

        done_outflow = self.cat_axis1(
            (reward[:, :-1], self.log((reward * (cfg['sample_flow_num'] + 1))[:, -1]).reshape(cfg['batch_size'], -1)))

        # loss
        critic_loss = self.mse(inflow * not_done, outflow * not_done) + \
                      self.mse(inflow * done_true, (done_outflow * done_true))
        critic_loss = self.mean(self.reduce_sum(critic_loss, 1))
        return critic_loss


class MSELoss(nn.Cell):
    """
    MSE loss function net
    """

    def __init__(self, network):
        super(MSELoss, self).__init__(auto_prefix=False)
        self.network = network
        self.mse = nn.MSELoss()
        self.cat_axis2 = P.Concat(axis=2)

    def construct(self, next_state, action, state):
        sa = self.cat_axis2((next_state, action))
        pre_state = self.network(sa)
        transaction_loss = self.mse(pre_state, state)
        return transaction_loss


class TrainOneStepCriticCell(nn.Cell):
    """Critic Network training package class."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCriticCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, inflow_state, inflow_action, outflow_state, outflow_action, not_done, done_true, reward):
        weights = self.weights
        loss = self.network(inflow_state, inflow_action, outflow_state, outflow_action, not_done, done_true, reward)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(inflow_state, inflow_action, outflow_state, outflow_action,
                                                 not_done, done_true, reward, sens)
        self.optimizer(grads)
        return loss


class TrainOneStepTransactionCell(nn.Cell):
    """Critic Network training package class."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepTransactionCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, next_state, action, state):
        weights = self.weights
        loss = self.network(next_state, action, state)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(next_state, action, state, sens)
        self.optimizer(grads)
        return loss


class CriticTrainNetWrapper(nn.Cell):
    """Wraps the model with optimizer"""

    def __init__(self, critic):
        super(CriticTrainNetWrapper, self).__init__(auto_prefix=False)
        self.critic = critic
        fmloss = FlowMatchLoss(self.critic)
        critic_optimizer = self.make_opt(fmloss, learning_rate=cfg['critic_lr'])
        self.critic_loss_train_net = TrainOneStepCriticCell(fmloss, critic_optimizer)

    def construct(self, inflow_state, inflow_action, outflow_state, outflow_action, not_done, done_true, reward):
        critic_loss = self.critic_loss_train_net(inflow_state, inflow_action, outflow_state, outflow_action,
                                                 not_done, done_true, reward)
        return critic_loss

    def make_opt(self, network, opt='adam', learning_rate=0.0001):
        """
        optimizer
        """
        if opt == 'momentum':
            optimizer = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=cfg['momentum'])
        elif opt == 'adam':
            optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)
        elif opt == 'sgd':
            optimizer = nn.SGD(network.trainable_params(), learning_rate=learning_rate, momentum=cfg['momentum'])
        else:
            optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)
        return optimizer


class TransactionTrainNetWrapper(nn.Cell):
    """Wraps the model with optimizer"""

    def __init__(self, transaction):
        super(TransactionTrainNetWrapper, self).__init__(auto_prefix=False)
        self.transaction = transaction
        mseloss = MSELoss(self.transaction)
        transaction_optimizer = self.make_opt(mseloss, learning_rate=cfg['transaction_lr'])
        self.transaction_loss_train_net = TrainOneStepTransactionCell(mseloss, transaction_optimizer)

    def construct(self, next_state, action, state):
        transaction_loss = self.transaction_loss_train_net(next_state, action, state)
        return transaction_loss

    def make_opt(self, network, opt='adam', learning_rate=0.0001):
        """
        optimizer
        """
        if opt == 'momentum':
            optimizer = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=cfg['momentum'])
        elif opt == 'adam':
            optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)
        elif opt == 'sgd':
            optimizer = nn.SGD(network.trainable_params(), learning_rate=learning_rate, momentum=cfg['momentum'])
        else:
            optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)
        return optimizer
