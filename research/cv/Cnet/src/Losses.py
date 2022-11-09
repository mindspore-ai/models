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
"""acr loss"""
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import numpy
import mindspore as ms


def distance_vectors_pairwise(anchor, positive, negative=None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = ops.ReduceSum()(anchor * anchor, axis=1)
    p_sq = ops.ReduceSum()(positive * positive, axis=1)

    eps = 1e-8
    d_a_p = ops.Rsqrt()(a_sq + p_sq -
                        2 * ops.ReduceSum()(anchor * positive, axis=1) + eps)
    if negative is not None:
        n_sq = ops.ReduceSum()(negative * negative, axis=1)
        d_a_n = ops.Rsqrt()(a_sq + n_sq -
                            2 * ops.ReduceSum()(anchor * negative, axis=1) +
                            eps)
        d_p_n = ops.Rsqrt()(p_sq + n_sq -
                            2 * ops.ReduceSum()(positive * negative, axis=1) +
                            eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = ops.ReduceSum()(anchor * anchor, axis=1).unsqueeze(-1)
    d2_sq = ops.ReduceSum()(positive * positive, axis=1).unsqueeze(-1)

    eps = 1e-6
    return ops.Sqrt()(
        (d1_sq.repeat(1, positive.shape[0]) +
         ops.Transpose(d2_sq.repeat(1, anchor.shape[0]), (1, 0)) -
         2.0 * ops.BatchMatMul()(anchor.unsqueeze(0), ops.Transpose()(positive, (1, 0)).unsqueeze(0)).squeeze(0)) + eps)


class Adaptive_Augular_Margin_Loss(nn.Cell):
    def __init__(self):
        super(Adaptive_Augular_Margin_Loss, self).__init__()

        self.matmul = ops.MatMul(transpose_a=False, transpose_b=True)
        self.eye = ops.Eye()
        self.mean = ops.ReduceMean(False)
        self.sum = ops.ReduceSum()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.ge = ops.GreaterEqual()
        self.cast = ops.Cast()

    def construct(self, anchor, positive, scale=1.0):
        eps = 1e-8
        # cosine matrix
        cos_matrix = self.matmul(anchor, positive) + eps
        cos_matrix = 1.0 - cos_matrix
        eye = self.eye(cos_matrix.shape[1], cos_matrix.shape[1], ms.float32)
        all_pos = numpy.diag(cos_matrix)
        dist_without_min_on_diag = cos_matrix + 10.0 * eye
        mask = (self.cast(self.ge(dist_without_min_on_diag, 0.05), ms.float32) -
                1.0) * (-1)
        mask = mask.astype(dist_without_min_on_diag.dtype) * 10.0
        dist_without_min_on_diag = dist_without_min_on_diag + mask
        neg1 = ops.ReduceMin(keep_dims=False)(dist_without_min_on_diag, 1)
        neg2 = ops.ReduceMin(keep_dims=False)(dist_without_min_on_diag, 0)
        all_neg = ops.Minimum()(neg1, neg2)

        loss = self.mean(self.log(1.0 + self.exp(scale * (all_pos - all_neg))))

        var_pos = self.mean((all_pos - self.mean(all_pos)).clip(0.0, None) ** 2)
        var_neg = self.mean((self.mean(all_neg) - all_neg).clip(0.0, None) ** 2)
        var_loss = (var_pos + var_neg) * 0.0
        loss += var_loss

        n = anchor.shape[0]
        symetric_loss = self.sum(
            (cos_matrix - cos_matrix.transpose()) ** 2) / (n * (n - 1))
        loss += symetric_loss
        return loss


def global_orthogonal_regularization(anchor, negative):
    dim = anchor.shape[1]
    neg_dis = ops.ReduceSum()(ops.Mul()(anchor, negative), 1)
    gor = ops.Pow()(ops.ReduceMean()(neg_dis), 2) + ops.clip_by_value(
        ops.ReduceMean()(ops.Pow()(neg_dis, 2)) - 1.0 / dim,
        clip_value_min=Tensor(0.0),
        clip_value_max=Tensor(float('inf')))

    return gor


def correlationPenaltyLoss(anchor):
    mean1 = ops.ReduceMean()(anchor)
    zeroed = anchor - mean1.expand_as(anchor)
    cor_mat = ops.BatchMatMul()(ops.Transpose()(zeroed, (1, 0)).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
    d = numpy.diag(numpy.diag(cor_mat))
    no_diag = cor_mat - d
    d_sq = no_diag * no_diag
    return ops.Sqrt()(d_sq.sum()) / anchor.shape[0]


class Losses(nn.Cell):
    def __init__(self, args):
        super(Losses, self).__init__()
        self.criterion = Adaptive_Augular_Margin_Loss()
        self.loss = args.loss
        self.decor = args.decor
        self.gor = args.gor
        self.alpha = args.alpha
        self.scale = args.scale

    def construct(self, out_a, out_p, out_n):
        loss = self.criterion(out_a, out_p, scale=self.scale)
        if self.decor:
            loss += correlationPenaltyLoss(out_a)

        if self.gor:
            loss += self.alpha * global_orthogonal_regularization(out_a, out_n)
        return loss
