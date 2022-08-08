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

import mindspore as ms
from mindspore import Tensor, ops

def edge_accuracy(preds: Tensor, target: Tensor) -> float:
    """
    Compute the accuracy of edge prediction (relation reconstruction).

    Parameters
    ----------
    preds : Tensor
        [E, batch, K], probability distribution of K types of relation for each edge.
    target : Tensor
        [batch, E], ground truth relations.

    Returns
    -------
    float
        accuracy edge prediction (relation reconstruction).

    """
    preds = preds.argmax(-1)
    correct = ops.equal(preds.T, target).astype("int32").sum()
    return correct / Tensor(target.size, ms.float32)


def cross_entropy(p: Tensor, q: Tensor, eps: float = 1e-16) -> Tensor:
    """
    Compute the negative cross-entropy between p and q, i.e., CE[p||q].
    """
    return (p * ops.Log()(ops.ReLU()(q) + eps)).sum()


def kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-16) -> Tensor:
    """
    Compute the KL-divergence between p and q, i.e., KL[p||q].
    """
    return cross_entropy(p, p, eps) - cross_entropy(p, q, eps)


def nll_gaussian(preds: Tensor, target: Tensor, variance: float) -> Tensor:
    """
    Compute the negative log-likelihood of a Gaussian distribution with a fixed variance.

    Parameters
    ----------
    preds : Tensor
        [batch, steps, node, dim].
    target : Tensor
        [batch, steps, node, dim].
    variance : float
        a fixed variance.

    Returns
    -------
    Tensor
        negative log-likelihood: [steps, dim].

    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    return neg_log_p.sum() / (target.shape[0] * target.shape[2])
