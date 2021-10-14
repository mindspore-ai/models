# Copyright 2021 Huawei Technologies Co., Ltd
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
Evaluation metrics
"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore import Tensor


def accuracy(preds, targets):
    """
    Calculate the accuracy between predictions and targets

    Args:
        preds(Tensor): predictions
        targets(Tensor): ground truth

    Returns:
        accuracy: defined as 1 - (norm(targets - preds) / norm(targets))
    """
    return 1 - np.linalg.norm(targets.asnumpy() - preds.asnumpy(), 'fro') / np.linalg.norm(targets.asnumpy(), 'fro')


def r2(preds, targets):
    """
    Calculate R square between predictions and targets

    Args:
        preds(Tensor): predictions
        targets(Tensor): ground truth

    Returns:
        R square: coefficient of determination
    """
    return (1 - P.ReduceSum()((targets - preds) ** 2) / P.ReduceSum()((targets - P.ReduceMean()(preds)) ** 2)).asnumpy()


def explained_variance(preds, targets):
    """
    Calculate the explained variance between predictions and targets

    Args:
        preds(Tensor): predictions
        targets(Tensor): ground truth

    Returns:
        Var: explained variance
    """
    return (1 - (targets - preds).var() / targets.var()).asnumpy()


def evaluate_network(net, max_val, eval_inputs, eval_targets):
    """
    Evaluate the performance of network
    """
    eval_inputs = Tensor(eval_inputs, mstype.float32)
    eval_preds = net(eval_inputs)
    eval_targets = Tensor(eval_targets, mstype.float32)
    eval_targets = eval_targets.reshape((-1, eval_targets.shape[2]))

    rmse = P.Sqrt()(nn.MSELoss()(eval_preds, eval_targets)).asnumpy()
    mae = nn.MAELoss()(eval_preds, eval_targets).asnumpy()
    acc = accuracy(eval_preds, eval_targets)
    r_2 = r2(eval_preds, eval_targets)
    var = explained_variance(eval_preds, eval_targets)

    print("=====Evaluation Results=====")
    print('RMSE:', '{:.6f}'.format(rmse * max_val))
    print('MAE:', '{:.6f}'.format(mae * max_val))
    print('Accuracy:', '{:.6f}'.format(acc))
    print('R2:', '{:.6f}'.format(r_2))
    print('Var:', '{:.6f}'.format(var))
    print("============================")
