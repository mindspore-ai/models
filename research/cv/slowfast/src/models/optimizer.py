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

"""Optimizer."""

import mindspore.nn as nn
import src.utils.lr_policy as lr_policy


def construct_optimizer(model, steps_per_epoch, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []

    for x in model.trainable_params():
        if x.name.endswith('.gamma') or x.name.endswith('.beta'):
            bn_parameters.append(x)
        else:
            non_bn_parameters.append(x)

    optim_params = [
        {"params": bn_parameters, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ]

    # generate lrs
    lrs = []
    for cur_epoch in range(cfg.SOLVER.MAX_EPOCH):
        for cur_step in range(steps_per_epoch):
            lrs.append(get_epoch_lr(cur_epoch + float(cur_step) / steps_per_epoch, cfg))

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return nn.SGD(
            optim_params,
            learning_rate=lrs,
            momentum=cfg.SOLVER.MOMENTUM,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    raise NotImplementedError(
        "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)
