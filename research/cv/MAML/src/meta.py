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
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import load_param_into_net
from src.crossentropy import CrossEntropy
from src.learner import Learner


class Meta(nn.Cell):
    """
    Meta Learner
    """

    def __init__(self, args, config, param=None):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        # self.meta_lr = init_lr_scheduler(args)
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.sens = 1

        self.outer_net = Learner(config, args.imgc, args.imgsz)
        self.inner_net = Learner(config, args.imgc, args.imgsz)

        if param is not None:
            load_param_into_net(self.outer_net, param)
        for _, cell in self.outer_net.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.gamma.requires_grad = False
        for _, cell in self.inner_net.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.gamma.requires_grad = False
        # self.outer_params = ms.ParameterTuple(self.outer_net.trainable_params() + self.outer_net.untrainable_params())
        # self.inner_params = ms.ParameterTuple(self.inner_net.trainable_params() + self.inner_net.untrainable_params())

        self.outer_params = ms.ParameterTuple(self.outer_net.trainable_params())
        self.inner_params = ms.ParameterTuple(self.inner_net.trainable_params())

        self.moving_outer_params = ms.ParameterTuple(self.outer_net.untrainable_params())
        self.moving_inner_params = ms.ParameterTuple(self.inner_net.untrainable_params())
        self.inner_params_trainable = ms.ParameterTuple(self.inner_net.trainable_params())

        self.ce = CrossEntropy()
        self.inner_netwithloss = nn.WithLossCell(self.inner_net, self.ce)
        self.outer_netwithloss = nn.WithLossCell(self.outer_net, self.ce)

        self.meta_optim = nn.AdamWeightDecay(params=self.outer_net.trainable_params(), learning_rate=self.meta_lr,
                                             weight_decay=1e-4)

        self.argmax = ops.Argmax(axis=1)
        self.reducesum = ops.ReduceSum(keep_dims=True)
        self.softmax = nn.Softmax(axis=1)
        self.zeroslike = ops.ZerosLike()
        self.print = ops.Print()
        self.weights_mask = [True, True, False, True, True, False, True, True, False, True, True, False, True, True]
        self.grad = C.GradOperation(get_by_list=True)

    def construct(self, x_spt, y_spt, x_qry, y_qry, is_train=True):
        """
        :param x_spt:   [b, setsz, c_, h, w] #32task 5class 5sample
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, _, _, _, _ = x_spt.shape
        querysz = x_qry.shape[1]
        grad_all = None
        loss = 0
        corrects = 0
        for i in range(task_num):
            # assign vlue of outer_net params to inner_net params
            self.assign_tuple(self.inner_params, self.outer_params)
            grad = self.grad(self.inner_netwithloss, self.inner_params_trainable)(x_spt[i], y_spt[i])
            fast_weights = self.get_fast_weights(self.inner_params_trainable, grad, self.update_lr)
            logits_q = self.inner_net(x_qry[i])
            pred_q = self.argmax(self.softmax(logits_q))
            correct = self.reducesum((pred_q == y_qry[i]).astype(ms.float32))
            self.assign_tuple(self.inner_params_trainable, fast_weights)

            for _ in range(1, self.update_step):
                self.update_step_func(x_spt[i], y_spt[i], x_qry[i], y_qry[i])

            # loss_q will be overwritten and just keep the loss_q on last update step.
            grad = self.grad(self.inner_netwithloss, self.inner_params_trainable)(x_spt[i], y_spt[i])
            fast_weights = self.get_fast_weights(self.inner_params_trainable, grad, self.update_lr)

            logits_q = self.inner_net(x_qry[i])
            pred_q = self.argmax(self.softmax(logits_q))
            correct = self.reducesum((pred_q == y_qry[i]).astype(ms.float32))

            self.assign_tuple(self.inner_params_trainable, fast_weights)

            logits_q = self.inner_net(x_qry[i])
            loss_q = self.ce(logits_q, y_qry[i])
            loss += loss_q

            pred_q = self.argmax(self.softmax(logits_q))
            correct = self.reducesum((pred_q == y_qry[i]).astype(ms.float32))
            corrects += correct

            grad = self.grad(self.inner_netwithloss, self.inner_params_trainable)(x_qry[i], y_qry[i])

            self.assign_tuple(self.moving_outer_params, self.moving_inner_params)

            if not i:
                grad_all = grad
            else:
                grad_all = gradops(grad_all, grad)

        # grad_all = gradops(grad_all, task_num
        loss = loss / task_num
        accs = corrects / (querysz * task_num)
        if is_train:
            loss = F.depend(loss, self.meta_optim(grad_all))
        return loss, accs

    def assign_tuple(self, param, value):
        """assign params from tuple(value) to tuple(param)"""
        for i in range(len(param)):
            ops.assign(param[i], value[i])
        return param

    def get_fast_weights(self, param, grad, lr):
        """fast_weights = param - lr * grad"""
        fast_weight = []
        for i in range(len(param)):
            if self.weights_mask[i]:
                fast_weight.append(param[i] - self.update_lr * grad[i])
            else:
                fast_weight.append(param[i])
        return fast_weight

    def update_step_func(self, x_spt, y_spt, x_qry, y_qry):
        grad = self.grad(self.inner_netwithloss, self.inner_params_trainable)(x_spt, y_spt)
        fast_weights = self.get_fast_weights(self.inner_params_trainable, grad, self.update_lr)
        F.depend(grad, fast_weights)
        logits_q = self.inner_net(x_qry)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        pred_q = self.argmax(self.softmax(logits_q))
        correct = self.reducesum((pred_q == y_qry).astype(ms.float32))
        self.assign_tuple(self.inner_params_trainable, fast_weights)
        return correct


def gradops(tup1, tup2):
    t = ()
    for k in range(len(tup1)):
        t = t + (tup1[k] + tup2[k],)
    return t


def weight2dict(para, weight):
    pk = para.keys()
    w = weight.copy()
    for ind, name in enumerate(pk):
        if (name.endswith('moving_mean') or name.endswith('moving_variance')):
            w.insert(ind, para[name])
        else:
            w[ind] = ms.Parameter(w[ind])

    return dict(zip(pk, w))


def init_lr_scheduler(opt):
    '''
    Initialize the learning rate scheduler
    '''
    milestone = [25, 70, 100, 200, opt.epoch]
    lr0 = opt.meta_lr
    bl = list(np.logspace(0, len(milestone) - 1, len(milestone), base=opt.lr_scheduler_gamma))
    lr = [lr0 * b for b in bl]
    lr_epoch = nn.piecewise_constant_lr(milestone, lr)
    return lr_epoch
