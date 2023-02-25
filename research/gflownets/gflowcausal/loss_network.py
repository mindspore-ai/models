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
from mindspore import nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C


class FlowMatchLoss(nn.Cell):
    """
    flow match loss function net
    """

    def __init__(self, network):
        super(FlowMatchLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loginf = Tensor([1000], mstype.float32)
        self.expend = P.ExpandDims()
        self.ms_range = P.Range()
        self.zeros = P.Zeros()
        self.log = P.Log()
        self.exp = P.Exp()
        self.pow = P.Pow()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.cat_axis1 = P.Concat(axis=1)
        self.mean = P.ReduceMean(keep_dims=False)
        self.print = P.Print()

    def construct(self, parents, actions, reward, parents_sub, done, parent_range):
        parents_action = self.network(parents)[parent_range, actions]
        next_q = self.network(parents_sub)
        # cal input flow
        in_flow = self.log(self.zeros(parents_sub.shape[0], mstype.float32) + self.exp(parents_action))
        # cal output flow
        next_qd = self.expend((1 - done), 1) * next_q + self.expend((done), 1) * -self.loginf
        out_flow = self.log(self.reduce_sum(self.exp(self.cat_axis1([self.expend(self.log(reward), 1), next_qd])), 1))
        # loss
        loss = self.mean(self.pow(in_flow - out_flow, 2), 0)
        return loss


class TrainOneStepCell(nn.Cell):
    """Network training package class."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, parents, actions, reward, parents_sub, done, parent_range):
        weights = self.weights
        loss = self.network(parents, actions, reward, parents_sub, done, parent_range)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(parents, actions, reward, parents_sub, done, parent_range, sens)
        self.optimizer(grads)
        return loss


class TrainNetWrapper(nn.Cell):
    """Wraps the model with optimizer"""

    def __init__(self, network, args):
        super(TrainNetWrapper, self).__init__(auto_prefix=False)
        self.network = network
        fmloss = FlowMatchLoss(self.network)
        optimizer = self.make_opt(fmloss, args)
        self.loss_train_net = TrainOneStepCell(fmloss, optimizer)

    def construct(self, parents, actions, reward, parents_sub, done, parent_range):
        loss = self.loss_train_net(parents, actions, reward, parents_sub, done, parent_range)
        return loss

    def make_opt(self, network, args):
        """
        optimizer
        """
        if args.opt == 'momentum':
            optimizer = nn.Momentum(network.trainable_params(), learning_rate=args.learning_rate,
                                    momentum=args.momentum)
        elif args.opt == 'adam':
            optimizer = nn.Adam(network.trainable_params(),
                                learning_rate=args.learning_rate,
                                beta1=args.adam_beta1,
                                beta2=args.adam_beta2,
                                weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            optimizer = nn.SGD(network.trainable_params(), learning_rate=args.learning_rate, momentum=args.momentum)
        else:
            optimizer = nn.Adam(network.trainable_params(), learning_rate=args.learning_rate)
        return optimizer
