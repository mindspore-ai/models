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
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import numpy
import mindspore as ms


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    expand_dims = ops.ExpandDims()
    op_sum = ops.ReduceSum()
    d1_sq = expand_dims(op_sum(anchor * anchor, 1), -1)
    d2_sq = expand_dims(op_sum(positive * positive, 1), -1)

    eps = 1e-6
    return ops.Sqrt()((numpy.tile(d1_sq, (1, positive.shape[0])) +
                       numpy.tile(d2_sq, (1, anchor.shape[0])).T -
                       2.0 * ops.BatchMatMul()(expand_dims(anchor, 0), expand_dims(positive.T, 0)).squeeze(0)) + eps)


def loss_recon(x, x_recon):
    loss = (x - x_recon) ** 2
    return loss.mean()


def loss_HardNet(anchor, positive, anchor_swap=False, anchor_ave=False, margin=1.0, batch_reduce='min',
                 loss_type="triplet_margin"):
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = ops.Eye()(dist_matrix.shape[1], dist_matrix.shape[1], ms.float32)
    pos1 = numpy.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10
    mask = (ops.Cast()(ops.GreaterEqual()(dist_without_min_on_diag, 0.008), ms.float32) - 1.0) * (-1)
    mask = mask.astype(dist_without_min_on_diag.dtype) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask
    min_neg = ops.ArgMinWithValue(1)(dist_without_min_on_diag)[1]
    if anchor_swap:
        min_neg2 = ops.ArgMinWithValue(0)(dist_without_min_on_diag)[1]
        min_neg = ops.Minimum()(min_neg, min_neg2)
    pos = pos1
    loss = ops.clip_by_value(margin + pos - min_neg,
                             clip_value_min=0.0,
                             clip_value_max=(margin + pos - min_neg).max())

    loss = ops.ReduceMean()(loss)
    return loss


def element_neig_rank_loss(features, embedding, tem):
    softmax = ops.Softmax(0)
    log = ops.Log()
    matmul = ops.MatMul(transpose_a=False, transpose_b=True)
    cosine_distance_l = 0.5 * (1 + matmul(features, features))
    cosine_distance_h = 0.5 * (1 + matmul(embedding, embedding))

    W_h0 = softmax(cosine_distance_h / tem).T
    W_l0 = softmax(cosine_distance_l / tem).T
    cross_loss0 = W_h0 * (log(W_h0) - log(W_l0))

    knowledge_loss = cross_loss0.mean()
    return knowledge_loss


class Losses(nn.Cell):
    def __init__(self, args):
        super(Losses, self).__init__()
        self.margin = args.margin
        self.anchorave = args.anchorave
        self.anchorswap = args.anchorswap
        self.batch_reduce = args.batch_reduce
        self.loss_type = args.loss
        self.reconw = args.reconw
        self.weight = args.weight
        self.temp = args.temp

    def construct(self, opt_img, sar_img, opt_e, sar_e, opt_lower_nor, sar_lower_nor, opt_recon, sar_recon):
        loss_matching = loss_HardNet(opt_e, sar_e, margin=self.margin, anchor_swap=self.anchorswap,
                                     anchor_ave=self.anchorave, batch_reduce=self.batch_reduce,
                                     loss_type=self.loss_type)
        loss_recon_opt = self.reconw * loss_recon(opt_img, opt_recon)
        loss_recon_sar = self.reconw * loss_recon(sar_img, sar_recon)

        loss_element_rank_neig_sys_opt = self.weight * element_neig_rank_loss(opt_lower_nor, opt_e, self.temp)
        loss_element_rank_neig_sys_sar = self.weight * element_neig_rank_loss(sar_lower_nor, sar_e, self.temp)

        loss_element_rank_neig_sys_opt_cross1 = self.weight * element_neig_rank_loss(sar_lower_nor, opt_e, self.temp)
        loss_element_rank_neig_sys_sar_cross2 = self.weight * element_neig_rank_loss(opt_lower_nor, sar_e, self.temp)

        loss_element_rank_neig_sys_cross3 = self.weight * element_neig_rank_loss(opt_e, sar_e, self.temp)
        loss_element_rank_neig_sys_cross4 = self.weight * element_neig_rank_loss(opt_lower_nor, sar_lower_nor,
                                                                                 self.temp)
        loss = loss_matching + loss_recon_opt + loss_recon_sar + loss_element_rank_neig_sys_opt + \
               loss_element_rank_neig_sys_sar + loss_element_rank_neig_sys_opt_cross1 + \
               loss_element_rank_neig_sys_sar_cross2 + loss_element_rank_neig_sys_cross3 + \
               loss_element_rank_neig_sys_cross4

        return loss
