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
"""
Re-ranking function

Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.

CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url: http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf

Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""
import numpy as np


def k_reciprocal_neigh(initial_rank, i, k1):
    """ Obtaining k-reciprocal nearest neighbours """
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    Distances re-ranking

    Args:
        q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
        q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
        g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
        k1: parameter, the original paper is (k1=20, k2=6, lambda_value=0.3)
        k2: parameter, the original paper is (k1=20, k2=6, lambda_value=0.3)
        lambda_value: parameter, the original paper is (k1=20, k2=6, lambda_value=0.3)

    Returns:

    """
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = 2. - 2 * original_dist  # np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    v = np.zeros_like(original_dist).astype(np.float32)
    # top K1+1
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        v[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num]
    if k2 != 1:
        v_qe = np.zeros_like(v, dtype=np.float32)
        for i in range(all_num):
            v_qe[i, :] = np.mean(v[initial_rank[i, :k2], :], axis=0)
        v = v_qe
        del v_qe
    del initial_rank
    inv_index = []
    for i in range(all_num):
        inv_index.append(np.where(v[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        ind_non_zero = np.where(v[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_non_zero]
        for j in range(len(ind_non_zero)):
            temp_min[0, ind_images[j]] = temp_min[0, ind_images[j]] + np.minimum(
                v[i, ind_non_zero[j]],
                v[ind_images[j], ind_non_zero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del v
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
