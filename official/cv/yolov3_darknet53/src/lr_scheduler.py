# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Learning rate scheduler."""
import math
from collections import Counter

import numpy as np


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """
    Warmup step learning rate.

    We use warmup step to optimize learning rate. Learning rate will increase from
    0 to the learning rate you set by
    linear_warmup_lr(initlr + \\frac{currentstep}{warmupstep}\\times (baselr - initlr)).
    After the increasing step, it will drop by
    polynomial(lr\\times gamma^{stepscounter_i}, stepcounter is the number of steps in
    this epoch).

    Args:
        lr(float): The learning rate you set.
        lr_epochs(list): Index of the epoch which leads a decay of learning rate.
        steps_per_epoch(int): Steps in one epoch.
        warmup_epochs(int): Index of the epoch which ends the warm_up step.
        max_epoch(int): Numbers of epochs.
        gamma(float): Parameter in decay function. Default:0.1.

    Returns:
        ndarray, learning rate of each step.

    Examples:
        >>> warmup_step_lr(0.01, [1,3,5,7,9], 1000, 5, 10)
        <<< array([2.e-06, 4.e-06, 6.e-06, ..., 1.e-05, 1.e-05, 1.e-05], dtype=float32)
    """
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma**milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def multi_step_lr(lr, milestones, steps_per_epoch, max_epoch, gamma=0.1):
    """
    Multi step learning rate.

    We use multi step to optimize learning rate. Learning rate will drop from the
    lr you set at milestone by
    polynomial(lr\\times gamma^{stepscounter_i}, stepcounter is the number of steps
    in this epoch).

    Args:
        lr(float): The learning rate you set.
        lr_epochs(list): Index of the epoch which leads a decay of learning rate.
        steps_per_epoch(int): Steps in one epoch.
        max_epoch(int): Numbers of epochs.
        gamma(float): Parameter in decay function. Default:0.1.

    Returns:
        ndarray, learning rate of each step.

    Examples:
        >>> multi_step_lr(0.01, [1,3,5,7,9], 1000, 10)
        <<< array([1.e-02, 1.e-02, 1.e-02, ..., 1.e-07, 1.e-07, 1.e-07], dtype=float32)
    """
    return warmup_step_lr(lr, milestones, steps_per_epoch, 0, max_epoch, gamma=gamma)


def step_lr(lr, epoch_size, steps_per_epoch, max_epoch, gamma=0.1):
    """
    Step drop learning rate.

    We use step drop to optimize learning rate.Learning rate will drop from the lr you
    set each epoch_size by
    polynomial(lr\\times gamma^{stepscounter_i}, stepcounter is the number of steps
    in this epoch).

    Args:
        lr(float): The learning rate you set.
        epoch_size(int): Numbers of epochs one decay.
        steps_per_epoch(int): Steps in one epoch.
        max_epoch(int): Numbers of epochs.
        gamma(float): Parameter in decay function. Default:0.1.

    Returns:
        ndarray, learning rate of each step.

    Examples:
        >>> step_lr(0.01, 5, 1000, 10)
        <<< array([0.01 , 0.01 , 0.01 , ..., 0.001, 0.001, 0.001], dtype=float32)
    """
    lr_epochs = []
    for i in range(1, max_epoch):
        if i % epoch_size == 0:
            lr_epochs.append(i)
    return multi_step_lr(lr, lr_epochs, steps_per_epoch, max_epoch, gamma=gamma)


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """Cosine annealing learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi*last_epoch / T_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def get_lr(args):
    """generate learning rate."""
    if args.lr_scheduler == 'exponential':
        lr = warmup_step_lr(args.lr,
                            args.lr_epochs,
                            args.steps_per_epoch,
                            args.warmup_epochs,
                            args.max_epoch,
                            gamma=args.lr_gamma,
                            )
    elif args.lr_scheduler == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(args.lr,
                                        args.steps_per_epoch,
                                        args.warmup_epochs,
                                        args.max_epoch,
                                        args.T_max,
                                        args.eta_min)
    else:
        raise NotImplementedError("args.lr_scheduler only support 'exponential' and 'cosine_annealing'"
                                  ", {} is not supported".format(args.lr_scheduler))
    return lr
