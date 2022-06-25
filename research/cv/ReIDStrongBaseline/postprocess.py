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
"""post process for 310 inference"""

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Train StrongBaseline')
parser.add_argument('--q_feature', type=str, default='')
parser.add_argument('--q_pid', type=str, default='')
parser.add_argument('--q_camid', type=str, default='')
parser.add_argument('--g_feature', type=str, default='')
parser.add_argument('--g_pid', type=str, default='')
parser.add_argument('--g_camid', type=str, default='')

parser.add_argument('--train_url', type=str, default='log')
parser.add_argument('--reranking', type=lambda x: x.lower() == 'true', default=True, help='re_rank')
parser.add_argument('--test_distance', type=str, default='global_local', help='test distance type')
parser.add_argument('--unaligned', action='store_true')
args = parser.parse_args()

def get_query(feature_file, pid_file, camid_file):
    """get query data"""
    qf, q_pids, q_camids = [], [], []
    openfilef = open(feature_file, 'r')
    for line in openfilef.readlines():
        temp = float(line)
        qf.append(temp)
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
    #add norm
    c = np.linalg.norm(qf, ord=2, axis=1, keepdims=True)
    qf = qf/c
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))
    return qf, q_pids, q_camids

def get_gallery(feature_file, pid_file, camid_file):
    """get gallery data"""
    gf, g_pids, g_camids = [], [], []
    openfilef = open(feature_file, 'r')
    for line in openfilef.readlines():
        temp = float(line)
        gf.append(temp)
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
    #add norm
    c = np.linalg.norm(gf, ord=2, axis=1, keepdims=True)
    gf = gf/c
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))
    return gf, g_pids, g_camids


def eval_func(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = dist_mat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(dist_mat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    print("==============================compute cmc curve for each query")
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


if __name__ == '__main__':
    sqf, sq_pids, sq_camids = get_query(args.q_feature, args.q_pid, args.q_camid)
    sgf, sg_pids, sg_camids = get_gallery(args.g_feature, args.g_pid, args.g_camid)
    m, n = sqf.shape[0], sgf.shape[0]
    distmat = np.power(sqf, 2).sum(axis=1, keepdims=True).repeat(n, axis=1) + \
        np.power(sgf, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
    distmat = 1 * distmat - 2 * np.dot(sqf, sgf.transpose())
    r, m_ap = eval_func(distmat, sq_pids, sg_pids, sq_camids, sg_camids)
    s = 'After BNNeck'
    print(f'[INFO] {s}')
    print(
        '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
        )
    )
