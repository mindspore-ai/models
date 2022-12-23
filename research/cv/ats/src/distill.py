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

"""Distill class."""

import mindspore
from mindspore import ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import save_checkpoint

from src.utils import count_acc, Averager
from src.utils import append_to_logs
from src.utils import format_logs
from src.utils import get_lr


class KLTSLoss(nn.LossBase):
    def __init__(self, t_tau, s_tau, reduction="sum"):
        super(KLTSLoss, self).__init__(reduction)
        self.log_softmax = ops.LogSoftmax(axis=1)
        self.softmax = ops.Softmax(axis=1)
        self.t_tau = t_tau
        self.s_tau = s_tau

    def construct(self, s_logits, t_logits):
        p_t = self.softmax(t_logits / self.t_tau)
        p_s = self.log_softmax(s_logits / self.s_tau)
        loss = -1.0 * (p_t * p_s).sum(axis=-1).mean()
        return loss

    def forward(self):
        print("Not implemented...", self.t_tau)
        return self.t_tau


class KLATSLoss(nn.LossBase):
    def __init__(self, tp_tau, t_tau, s_tau, reduction="sum"):
        super(KLATSLoss, self).__init__(reduction)
        self.log_softmax = ops.LogSoftmax(axis=1)
        self.softmax = ops.Softmax(axis=1)
        self.tp_tau = tp_tau
        self.t_tau = t_tau
        self.s_tau = s_tau

    def construct(self, s_logits, t_logits, labels):
        taus = self.t_tau * ops.OnesLike()(t_logits)
        inds = mindspore.numpy.arange(len(s_logits))
        taus[inds, labels] = self.tp_tau

        p_t = self.softmax(t_logits / taus)
        p_s = self.log_softmax(s_logits / self.s_tau)
        loss = -1.0 * (p_t * p_s).sum(axis=-1).mean()
        return loss

    def forward(self):
        print("Not implemented...", self.tp_tau)
        return self.tp_tau


class KDTSLossCell(nn.Cell):
    def __init__(self, student_net, t_tau, s_tau, lamb):
        super(KDTSLossCell, self).__init__()
        self.student_net = student_net
        self.t_tau = t_tau
        self.s_tau = s_tau
        self.lamb = lamb
        self.kl_loss = KLTSLoss(
            t_tau=t_tau, s_tau=s_tau
        )
        self.ce_loss = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='mean'
        )

    def construct(self, t_logits, xs, ys):
        s_logits = self.student_net(xs)
        ce_loss = self.ce_loss(s_logits, ys)
        kl_loss = self.kl_loss(
            s_logits=s_logits, t_logits=t_logits
        )
        return (1.0 - self.lamb) * ce_loss \
            + self.lamb * self.s_tau * self.s_tau * kl_loss

    @property
    def backbone_network(self):
        return self.student_net


class KDATSLossCell(nn.Cell):
    def __init__(self, student_net, tp_tau, t_tau, s_tau, lamb):
        super(KDATSLossCell, self).__init__()
        self.student_net = student_net
        self.tp_tau = tp_tau
        self.t_tau = t_tau
        self.s_tau = s_tau
        self.lamb = lamb
        self.kl_loss = KLATSLoss(
            tp_tau=tp_tau, t_tau=t_tau, s_tau=s_tau
        )
        self.ce_loss = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='mean'
        )

    def construct(self, t_logits, xs, ys):
        s_logits = self.student_net(xs)
        ce_loss = self.ce_loss(s_logits, ys)
        kl_loss = self.kl_loss(
            s_logits=s_logits, t_logits=t_logits, labels=ys
        )
        return (1.0 - self.lamb) * ce_loss \
            + self.lamb * self.s_tau * self.s_tau * kl_loss

    @property
    def backbone_network(self):
        return self.student_net


class KDTrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(KDTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        grads = self.grad(self.network, weights)(*inputs)
        grads = self.grad_reducer(grads)
        return loss, self.optimizer(grads)

    def forward(self):
        print("Not implemented...", self.grad)
        return self.grad


class Distill():
    def __init__(self, train_set, test_set, model, tmodel, dargs):
        self.train_set = train_set
        self.test_set = test_set
        self.model = model
        self.tmodel = tmodel
        self.dargs = dargs

        if dargs.kd_way == "TS":
            self.net_with_KD = KDTSLossCell(
                student_net=self.model,
                t_tau=dargs.t_tau,
                s_tau=dargs.s_tau,
                lamb=dargs.lamb
            )
        elif dargs.kd_way == "ATS":
            self.net_with_KD = KDATSLossCell(
                student_net=self.model,
                tp_tau=dargs.tp_tau,
                t_tau=dargs.t_tau,
                s_tau=dargs.s_tau,
                lamb=dargs.lamb
            )
        else:
            raise ValueError("No such kd_way: {}".format(dargs.kd_way))

        self.step_size = self.train_set.get_dataset_size()
        lrs = Tensor(get_lr(
            lr_init=0.001, lr_max=dargs.lr, warmup_epochs=3,
            total_epochs=dargs.epoches, steps_per_epoch=self.step_size
        ))

        self.optimizer = nn.Momentum(
            model.get_parameters(),
            learning_rate=lrs,
            momentum=dargs.momentum
        )
        self.train_step = KDTrainStep(
            self.net_with_KD, self.optimizer
        )

        self.logs = {
            "EPOCHS": [],
            "LOSSES": [],
            "TrACCS": [],
            "TeACCS": [],
        }

    def main(self):
        for epoch in range(1, self.dargs.epoches + 1):
            ce_loss, tr_acc = self.train(
                model=self.model,
                tmodel=self.tmodel,
                dset=self.train_set
            )
            te_acc = self.test(
                model=self.model,
                dset=self.test_set
            )
            print("[Epoch:{}] [Loss:{}] [TrAcc:{}] [TeAcc:{}]".format(
                epoch, ce_loss, tr_acc, te_acc
            ))

            # add to log
            self.logs["EPOCHS"].append(epoch)
            self.logs["LOSSES"].append(ce_loss)
            self.logs["TrACCS"].append(tr_acc)
            self.logs["TeACCS"].append(te_acc)

    def train(self, model, tmodel, dset):
        model.set_train()
        tmodel.set_train(False)

        avg_ce_loss = Averager()
        acc_avg = Averager()

        for batch in dset.create_dict_iterator():
            xs, ys = batch["xs"], batch["ys"]
            tlogits = tmodel(xs)

            self.train_step(tlogits, xs, ys)
            loss = self.net_with_loss(tlogits, xs, ys).asnumpy()
            avg_ce_loss.add(loss)

            logits = model(xs)
            acc = count_acc(logits, ys)
            acc_avg.add(acc)

        ce_loss = avg_ce_loss.item()
        tr_acc = acc_avg.item()
        return ce_loss, tr_acc

    def test(self, model, dset):
        print("Epoches: ", self.dargs.epoches)
        model.set_train(False)

        acc_avg = Averager()
        for batch in dset.create_dict_iterator():
            xs, ys = batch["xs"], batch["ys"]
            logits = model(xs)

            acc = count_acc(logits, ys)
            acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_ckpt(self, fpath):
        # save model
        save_checkpoint(self.model, fpath)
        print("Model saved in:", fpath)

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.dargs))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
