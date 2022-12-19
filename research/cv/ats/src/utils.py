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

"""Helper functions."""


import numpy as np


def append_to_logs(fpath, logs):
    with open(fpath, "a", encoding="utf-8") as fa:
        for log in logs:
            fa.write("{}\n".format(log))
        fa.write("\n")


def format_logs(logs):
    def formal_str(x):
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            return "{:.5f}".format(x)
        return str(x)

    logs_str = []
    for key, elems in logs.items():
        log_str = "[{}]: ".format(key)
        log_str += " ".join([formal_str(e) for e in elems])
        logs_str.append(log_str)
    return logs_str


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    preds = logits.argmax(axis=1).asnumpy()
    reals = label.asnumpy()
    return np.mean(preds == reals)


def _generate_poly_lr(lr_init, lr_max, total_steps, warmup_steps):
    lr_steps = []

    inc = 0
    if warmup_steps != 0:
        inc = (lr_max - lr_init) / warmup_steps

    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + inc * i
        else:
            t = 1.0 - (i - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_max * (t ** 2)

        if lr < 0.0:
            lr = 0.0

        lr_steps.append(lr)
    return lr_steps


def get_lr(lr_init, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    lr_steps = []
    lr_steps = _generate_poly_lr(lr_init, lr_max, steps_per_epoch * total_epochs, steps_per_epoch * warmup_epochs)
    lr_steps = np.array(lr_steps).astype(np.float32)
    return lr_steps
