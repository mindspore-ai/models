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
""" evalfunc.py """

import os
import time
import numpy as np
import psutil
import mindspore.ops as P

POOLDIM = 2048
KINDS = 4


def show_memory_info(hint=""):
    """
    Show memory info
    """
    pid = os.getpid()

    process = psutil.Process(pid)
    info = process.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")


def test(args, gallery, query, ngall, nquery,
         backbone, gallery_cam=None, query_cam=None):
    """
    test
    """
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    cat = P.Concat(1)

    gall_label = np.zeros((ngall,))
    query_label = np.zeros((nquery,))

    gall_feat_ob = np.zeros((ngall, POOLDIM * KINDS))  # 2048 x 4

    for (img, label) in gallery:
        feat_v_ob, feat_v_shared_ob, feat_i_ob, feat_i_shared_ob = backbone(img)

        size = int(feat_v_ob.shape[0])  # batch size
        ob = cat((feat_v_ob, feat_v_shared_ob, feat_i_ob, feat_i_shared_ob))

        gall_feat_ob[ptr: ptr + size, :] = ob.asnumpy()
        gall_label[ptr: ptr + size] = label.asnumpy()
        ptr = ptr + size
    print(f'Extracting Time :\t {time.time() - start:.3f}')

    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat_ob = np.zeros((nquery, POOLDIM * KINDS))

    for (img, label) in query:
        feat_v_ob, feat_v_shared_ob, feat_i_ob, feat_i_shared_ob = backbone(img)

        size = int(feat_v_ob.shape[0])
        ob = cat((feat_v_ob, feat_v_shared_ob, feat_i_ob, feat_i_shared_ob))

        query_feat_ob[ptr: ptr + size, :] = ob.asnumpy()

        query_label[ptr: ptr + size] = label.asnumpy()

        ptr = ptr + size
    print(f'Extracting Time :\t {time.time() - start:.3f}')

    # compute the similarity
    distmat_ob = np.matmul(query_feat_ob, np.transpose(gall_feat_ob))

    if args.dataset == "SYSU":
        cmc1, map1 = eval_sysu(-distmat_ob, query_label, gall_label, query_cam, gallery_cam)

    elif args.dataset == "RegDB":
        cmc1, map1 = eval_regdb(-distmat_ob, query_label, gall_label)

    return cmc1, map1


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # reference: Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_inp.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
        # Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        ap_ = tmp_cmc.sum() / num_rel
        all_ap.append(ap_)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    map_ = np.mean(all_ap)
    return new_all_cmc, map_


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """
    eval_regdb
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # reference: Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_inp.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference:
        # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        ap_ = tmp_cmc.sum() / num_rel
        all_ap.append(ap_)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    map_ = np.mean(all_ap)
    return all_cmc, map_
