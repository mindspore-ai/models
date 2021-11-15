# ------------------------------------------------------------------------------
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
'''
nms operation
'''
from __future__ import division
import numpy as np


delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
areaThres = 0  # 40 * 40.5
alpha = 0.1


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    '''
    oks_iou
    '''
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72,
                           .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    var = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / var / \
            ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    kpts = len(kpts_db)
    if kpts == 0:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten()
                     for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]],
                          areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def pose_nms(bboxes, bbox_scores, pose_preds, pose_scores):
    '''
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n,)
    pose_preds:     pose locations list (n, 17, 2)
    pose_scores:    pose scores list    (n, 17, 1)
    '''

    pose_scores[pose_scores == 0] = 1e-5

    final_result = []

    ori_bbox_scores = bbox_scores.copy()
    ori_pose_preds = pose_preds.copy()
    ori_pose_scores = pose_scores.copy()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = np.mean(pose_scores, axis=1)

    human_ids = np.arange(nsamples)
    # Do pPose-NMS
    pick = []
    merge_ids = []
    while human_scores.shape[0] != 0:
        # Pick the one with highest score
        pick_id = np.argmax(human_scores)
        pick.append(human_ids[pick_id])

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[pick_id]]
        simi = get_parametric_distance(
            pick_id, pose_preds, pose_scores, ref_dist)
        num_match_keypoints = PCK_match(
            pose_preds[pick_id], pose_preds, ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity

        delete_ids = np.arange(human_scores.shape[0])[(
            simi > gamma) | (num_match_keypoints >= matchThreds)]
        if delete_ids.shape[0] == 0:
            delete_ids = pick_id

        merge_ids.append(human_ids[delete_ids])
        pose_preds = np.delete(pose_preds, delete_ids, axis=0)
        pose_scores = np.delete(pose_scores, delete_ids, axis=0)
        human_ids = np.delete(human_ids, delete_ids)
        human_scores = np.delete(human_scores, delete_ids, axis=0)
        bbox_scores = np.delete(bbox_scores, delete_ids, axis=0)

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]

    for j in range(len(pick)):
        ids = np.arange(17)
        max_score = np.max(scores_pick[j, ids, 0])

        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])

        max_score = np.max(merge_score[ids])
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])

        if 1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres:
            continue

        final_result.append({
            'keypoints': merge_pose - 0.3,
            'kp_score': merge_score,
            'proposal_score': np.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score)
        })

    return final_result


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [17, 2]
        cluster_preds:  redundant poses         -- [n, 17, 2]
        cluster_scores: redundant poses score   -- [n, 17, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [17, 2]
        final_score:    merged score            -- [17]
    '''
    dist = np.sqrt(np.sum(
        np.power(ref_pose[np.newaxis, :] - cluster_preds, 2),
        axis=2
    ))

    kp_num = 17
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = np.zeros((kp_num, 2))
    final_score = np.zeros(kp_num)

    if cluster_preds.ndim == 2:
        cluster_preds = np.expand_dims(cluster_preds, axis=0)
        cluster_scores = np.expand_dims(cluster_scores, axis=0)
    if mask.ndim == 1:
        mask = np.expand_dims(mask, axis=0)

    # Weighted Merge
    masked_scores = np.multiply(
        cluster_scores, np.expand_dims(mask.astype(np.float32), axis=-1))
    normed_scores = masked_scores / np.sum(masked_scores, axis=0)

    final_pose = np.multiply(cluster_preds, np.tile(
        normed_scores, (1, 1, 2))).sum(axis=0)
    final_score = np.multiply(masked_scores, normed_scores).sum(axis=0)
    return final_pose, final_score


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    '''
    get parametric distance
    '''
    pick_preds = all_preds[i]

    pred_scores = keypoint_scores[i]
    dist = np.sqrt(np.sum(
        np.power(pick_preds[np.newaxis, :] - all_preds, 2),
        axis=2
    ))

    # Define a keypoints distance
    score_dists = np.zeros((all_preds.shape[0], 17), dtype=np.float32)
    keypoint_scores = np.squeeze(keypoint_scores)

    if keypoint_scores.ndim == 1:
        keypoint_scores = np.expand_dims(keypoint_scores, axis=0)
    if pred_scores.ndim == 1:
        pred_scores = np.expand_dims(pred_scores, axis=1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = np.tile(pred_scores, (1, all_preds.shape[0])).transpose(1, 0)

    point_dist = np.exp((-1) * dist / delta2)
    final_dist = np.sum(score_dists, axis=1) + mu * np.sum(point_dist, axis=1)

    return final_dist


def PCK_match(pick_pred, all_preds, ref_dist):
    '''
    PCK_match
    '''
    dist = np.sqrt(np.sum(
        np.power(pick_pred[np.newaxis, :] - all_preds, 2),
        axis=2
    ))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = np.sum(
        dist / ref_dist <= 1,
        axis=1
    )
    return num_match_keypoints
