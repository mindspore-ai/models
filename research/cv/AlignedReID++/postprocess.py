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
"""post process for 310 inference"""

import sys
import argparse
import os.path as osp
import numpy as np

from src.utils import Logger
from src.eval_metrics import evaluate

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
parser.add_argument('--q_feature', type=str, default='')
parser.add_argument('--q_localfeature', type=str, default='')
parser.add_argument('--q_pid', type=str, default='')
parser.add_argument('--q_camid', type=str, default='')
parser.add_argument('--g_feature', type=str, default='')
parser.add_argument('--g_localfeature', type=str, default='')
parser.add_argument('--g_pid', type=str, default='')
parser.add_argument('--g_camid', type=str, default='')
parser.add_argument('--train_url', type=str, default='log')
parser.add_argument('--reranking', type=lambda x: x.lower() == 'true', default=True, help='re_rank')
parser.add_argument('--test_distance', type=str, default='global_local', help='test distance type')
parser.add_argument('--unaligned', action='store_true')
args = parser.parse_args()

def get_query(feature_file, localfeature_file, pid_file, camid_file):
    """get query data"""
    qf, lqf, q_pids, q_camids = [], [], [], []
    openfilef = open(feature_file, 'r')
    for line in openfilef.readlines():
        temp = float(line)
        qf.append(temp)
    openfilel = open(localfeature_file, 'r')
    for line in openfilel.readlines():
        temp = float(line)
        lqf.append(temp)
    openfilep = open(pid_file, 'r')
    for line in openfilep.readlines():
        temp = int(line)
        q_pids.append(temp)
    openfilec = open(camid_file, 'r')
    for line in openfilec.readlines():
        temp = int(line)
        q_camids.append(temp)
    qf = np.array(qf)
    qf = qf.reshape(-1, 2048)
    lqf = np.array(lqf)
    lqf = lqf.reshape((-1, 2048, 8))
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))
    return qf, lqf, q_pids, q_camids

def get_gallery(feature_file, localfeature_file, pid_file, camid_file):
    """get gallery data"""
    gf, lgf, g_pids, g_camids = [], [], [], []
    openfilef = open(feature_file, 'r')
    for line in openfilef.readlines():
        temp = float(line)
        gf.append(temp)
    openfilel = open(localfeature_file, 'r')
    for line in openfilel.readlines():
        temp = float(line)
        lgf.append(temp)
    openfilep = open(pid_file, 'r')
    for line in openfilep.readlines():
        temp = int(line)
        g_pids.append(temp)
    openfilec = open(camid_file, 'r')
    for line in openfilec.readlines():
        temp = int(line)
        g_camids.append(temp)
    gf = np.array(gf)
    gf = gf.reshape(-1, 2048)
    lgf = np.array(lgf)
    lgf = lgf.reshape((-1, 2048, 8))
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))
    return gf, lgf, g_pids, g_camids

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    """compute reranking when test"""
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    if only_local:
        original_dist = local_distmat
    else:
        feat = np.concatenate([probFea, galFea])
        feat2 = feat

        feat = np.power(feat, 2)
        feat = np.sum(feat, axis=1, keepdims=True)
        feat = np.broadcast_to(feat, (all_num, all_num))
        feat_transpose = np.transpose(feat, (1, 0))
        distmat = feat+feat_transpose

        feat2_transpose = np.transpose(feat2, (1, 0))
        opt1 = np.dot(feat2, feat2_transpose)
        distmat = distmat + opt1*(-2)

        original_dist = distmat
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            leng1 = len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index))
            if leng1 > 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def test(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids):
    """test"""
    ranks = [1, 5, 10, 20]

    m_qf = np.linalg.norm(x=qf, ord=2, axis=-1, keepdims=True)
    m_qf = np.broadcast_to(m_qf, qf.shape)
    qf = 1.*qf /(m_qf + 1e-12)
    m_gf = np.linalg.norm(x=gf, ord=2, axis=-1, keepdims=True)
    m_gf = np.broadcast_to(m_gf, gf.shape)
    gf = 1.*gf /(m_gf + 1e-12)

    m, n = qf.shape[0], gf.shape[0]
    distqf = np.power(qf, 2)
    distqf = np.sum(distqf, axis=1, keepdims=True)
    distqf = np.broadcast_to(distqf, (m, n))
    distgf = np.power(gf, 2)
    distgf = np.sum(distgf, axis=1, keepdims=True)
    distgf = np.broadcast_to(distgf, (n, m))

    transpose_distgf = np.transpose(distgf, (1, 0))
    distmat = distqf + transpose_distgf
    transpose_gf = np.transpose(gf, (1, 0))
    temp = np.matmul(qf, transpose_gf)
    temp = temp*(-2)
    distmat = distmat+temp  # global_distmat

    if not args.test_distance == 'global':
        from src.distance import low_memory_local_dist
        lqf = np.transpose(lqf, (0, 2, 1))
        lgf = np.transpose(lgf, (0, 2, 1))
        local_distmat = low_memory_local_dist(lqf, lgf, aligned=not args.unaligned) # local_distmat
        if args.test_distance == 'local':
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))

    if args.reranking:
        if args.test_distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf, lqf, aligned=not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf, lgf, aligned=not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3, local_distmat=local_dist, only_local=True)
            elif args.test_distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3, local_distmat=local_dist, only_local=False)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    return cmc[0]

if __name__ == '__main__':
    sys.stdout = Logger(osp.join(args.train_url, 'log_test.txt'))
    sqf, slqf, sq_pids, sq_camids = get_query(args.q_feature, args.q_localfeature, args.q_pid, args.q_camid)
    sgf, slgf, sg_pids, sg_camids = get_gallery(args.g_feature, args.g_localfeature, args.g_pid, args.g_camid)
    test(sqf, sgf, slqf, slgf, sq_pids, sg_pids, sq_camids, sg_camids)
