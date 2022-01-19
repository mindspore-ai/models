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
"""PWCNET learning rate scheduler."""
from collections import Counter
import numpy as np

class _WarmUp():
    """
    Basic class for warm up
    """
    def __init__(self, warmup_init_lr):
        self.warmup_init_lr = warmup_init_lr

    def get_lr(self):
        # Get learning rate during warmup
        raise NotImplementedError

class _LRScheduler():
    """
    Basic class for learning rate scheduler
    """
    def __init__(self, lr, max_epoch, steps_per_epoch):
        self.base_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = int(max_epoch * steps_per_epoch)

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

class _LinearWarmUp(_WarmUp):
    """
    Class for linear warm up
    """
    def __init__(self, lr, warmup_epochs, steps_per_epoch, warmup_init_lr=0):
        self.base_lr = lr
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)

        super(_LinearWarmUp, self).__init__(warmup_init_lr)

    def get_warmup_steps(self):
        return self.warmup_steps

    def get_lr(self, current_step):
        lr_inc = (float(self.base_lr) - float(self.warmup_init_lr)) / float(self.warmup_steps)
        lr = float(self.warmup_init_lr) + lr_inc * current_step
        return lr

class MultiStepLR(_LRScheduler):
    """
    Multi-step learning rate scheduler

    Decays the learning rate by gamma once the number of epoch reaches one of the milestones.

    Args:
        lr (float): Initial learning rate which is the lower boundary in the cycle.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
        steps_per_epoch (int): The number of steps per epoch to train for.
        max_epoch (int): The number of epochs to train for.
        warmup_epochs (int, optional): The number of epochs to Warmup. Default: 0

    Outputs:
        numpy.ndarray, shape=(1, steps_per_epoch*max_epoch)

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(lr=0.1, milestones=[30,80], gamma=0.1, steps_per_epoch=5000, max_epoch=90)
        >>> lr = scheduler.get_lr()
    """

    def __init__(self, lr, milestones, gamma, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(MultiStepLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            if i < warmup_steps:
                lr = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                if i % self.steps_per_epoch == 0 and cur_ep in self.milestones:
                    current_lr = current_lr * self.gamma
                lr = current_lr

            lr_each_step.append(lr)

        return np.array(lr_each_step).astype(np.float32)
