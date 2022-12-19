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

"""Classify class."""

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import save_checkpoint

from src.utils import count_acc, Averager
from src.utils import append_to_logs
from src.utils import format_logs
from src.utils import get_lr


class Classify():
    def __init__(self, train_set, test_set, model, cargs):
        self.train_set = train_set
        self.test_set = test_set
        self.model = model
        self.cargs = cargs

        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction="mean"
        )
        self.net_with_loss = nn.WithLossCell(
            self.model, self.loss_fn
        )

        self.step_size = self.train_set.get_dataset_size()
        lrs = Tensor(get_lr(
            lr_init=0.001, lr_max=cargs.lr, warmup_epochs=3,
            total_epochs=cargs.epoches, steps_per_epoch=self.step_size
        ))

        self.optimizer = nn.Momentum(
            model.get_parameters(),
            learning_rate=lrs,
            momentum=cargs.momentum
        )
        self.train_step = nn.TrainOneStepCell(
            self.net_with_loss, self.optimizer
        )

        self.logs = {
            "EPOCHS": [],
            "LOSSES": [],
            "TrACCS": [],
            "TeACCS": [],
        }

    def main(self):
        for epoch in range(1, self.cargs.epoches + 1):
            ce_loss, tr_acc = self.train(
                model=self.model,
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

    def train(self, model, dset):
        model.set_train()

        avg_ce_loss = Averager()
        acc_avg = Averager()

        for batch in dset.create_dict_iterator():
            xs, ys = batch["xs"], batch["ys"]
            self.train_step(xs, ys)
            loss = self.net_with_loss(xs, ys).asnumpy()
            avg_ce_loss.add(loss)

            logits = model(xs)
            acc = count_acc(logits, ys)
            acc_avg.add(acc)

        ce_loss = avg_ce_loss.item()
        tr_acc = acc_avg.item()
        return ce_loss, tr_acc

    def test(self, model, dset):
        print("Epoches: ", self.cargs.epoches)
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
        print("Model saved in: ", fpath)

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.cargs))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
