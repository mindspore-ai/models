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
"""Functions of optimizer"""

from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.optim.momentum import Momentum

from .schedulers import get_policy


def get_learning_rate(args, batch_num):
    """Get learning rate"""
    return get_policy(args.lr_scheduler)(args, batch_num)


def get_optimizer(args, model, batch_num):
    """Get optimizer for training"""
    print(f"=> When using train_wrapper, using optimizer {args.optimizer}")
    args.start_epoch = int(args.start_epoch)
    optim_type = args.optimizer.lower()
    params = get_param_groups(model)
    learning_rate = get_learning_rate(args, batch_num)
    step = int(args.start_epoch * batch_num)
    accumulation_step = int(args.accumulation_step)
    train_step = len(learning_rate)
    print(f"=> Get LR from epoch: {args.start_epoch}\n"
          f"=> Start step: {step}\n"
          f"=> Total step: {train_step}\n"
          f"=> Accumulation step:{accumulation_step}")
    if accumulation_step > 1:
        learning_rate = learning_rate * accumulation_step

    if optim_type == "momentum":
        optim = Momentum(
            params=params,
            learning_rate=learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif optim_type == "adamw":
        optim = AdamWeightDecay(
            params=params,
            learning_rate=learning_rate,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"optimizer {optim_type} is not supported")

    return optim


def get_param_groups(network):
    """ get param groups """
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith(".gamma") or parameter_name.endswith(".beta") or \
                parameter_name.endswith(".bias"):
            # Dense or Conv's weight using weight decay
            print(f"=> no decay {parameter_name}")
            no_decay_params.append(x)
        else:
            print(f"=> decay {parameter_name}")
            # all bias not using weight decay
            # bn weight bias not using weight decay, be carefully for now x not include LN
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]
