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
# This file was copied from project [sshaoshuai][https://github.com/sshaoshuai/PointRCNN]
"""Proposal Layer"""
import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor
from src.lib.utils.bbox_transform import decode_bbox_target
from src.lib.config import cfg
import src.lib.utils.kitti_utils as kitti_utils
import src.lib.utils.iou3d.iou3d_utils as iou3d_utils


class ProposalLayer(nn.Cell):
    """ProposalLayer class"""
    def __init__(self, mode='TRAIN'):
        super(ProposalLayer, self).__init__()
        self.mode = mode

        self.MEAN_SIZE = ms.Tensor.from_numpy(cfg.CLS_MEAN_SIZE[0])

    def construct(self, rpn_scores: Tensor, rpn_reg: Tensor, xyz: Tensor):
        """
        :param rpn_scores: (B, N)
        :param rpn_reg: (B, N, 8)
        :param xyz: (B, N, 3)
        :return bbox3d: (B, M, 7)
        """
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, 3),
                                       rpn_reg.view(-1, rpn_reg.shape[-1]),
                                       anchor_size=self.MEAN_SIZE,
                                       loc_scope=cfg.RPN.LOC_SCOPE,
                                       loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                       num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                       get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                       get_y_by_bin=False,
                                       get_ry_fine=False)  # (N, 7)
        proposals[:, 1] += proposals[:, 3] / 2  # set y as the center of bottom
        proposals = proposals.view(batch_size, -1, 7)

        scores = rpn_scores
        sort = ops.Sort(axis=1, descending=True)
        _, sorted_idxs = sort(scores)

        batch_size = scores.shape[0]

        ret_bbox3d = ms.numpy.zeros(
            (batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, 7), scores.dtype)
        ret_scores = ms.numpy.zeros(
            (batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N), scores.dtype)
        for k in range(batch_size):
            scores_single = scores[k]
            proposals_single = proposals[k]
            order_single = sorted_idxs[k]

            if cfg.TEST.RPN_DISTANCE_BASED_PROPOSE:
                scores_single, proposals_single = self.distance_based_proposal(
                    scores_single, proposals_single, order_single)
            else:
                scores_single, proposals_single = self.score_based_proposal(
                    scores_single, proposals_single, order_single)

            proposals_tot = proposals_single.shape[0]
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single

        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N  #9000
        pre_top_n_list = [
            0,
            int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)
        ]
        post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N  # 512
        post_top_n_list = [
            0,
            int(post_tot_top_n * 0.7),
            post_tot_top_n - int(post_tot_top_n * 0.7)
        ]

        scores_single_list, proposals_single_list = [], []

        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        dist = proposals_ordered[:, 2]
        first_mask = ops.logical_and((dist > nms_range_list[0]),
                                     (dist <= nms_range_list[1]))
        for i in range(1, len(nms_range_list)):
            # get proposal distance mask

            dist_mask: ms.Tensor = ops.logical_and(
                (dist > nms_range_list[i - 1]), (dist <= nms_range_list[i]))
            if dist_mask.sum() != 0:  #True
                # this area has points
                # reduce by mask

                _, idx = ops.Sort(descending=True)(dist_mask.astype(
                    ms.float32))
                # fetch pre nms top K
                cur_scores = scores_ordered[idx]
                cur_proposals = proposals_ordered[idx]

                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]

            else:
                assert i == 2, '%d' % i
                # this area doesn't have any points, so use rois of first area

                _, idx = ops.Sort(descending=True)(first_mask.astype(
                    ms.float32))
                cur_scores = scores_ordered[idx]
                cur_proposals = proposals_ordered[idx]

                # fetch top K of first area
                cur_scores = cur_scores[pre_top_n_list[i -
                                                       1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[
                    pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

            # oriented nms
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
            if cfg.RPN.NMS_TYPE == 'rotate':  #False
                keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores,
                                               cfg[self.mode].RPN_NMS_THRESH)
            elif cfg.RPN.NMS_TYPE == 'normal':  #True
                keep_idx = iou3d_utils.nms_normal_gpu(
                    boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            else:
                raise NotImplementedError

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])
        cat = ops.Concat(axis=0)
        scores_single = cat(scores_single_list)
        proposals_single = cat(proposals_single_list)
        return scores_single, proposals_single

    def score_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        # pre nms top K
        cur_scores = scores_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]

        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores,
                                       cfg[self.mode].RPN_NMS_THRESH)

        # Fetch post nms top k
        keep_idx = keep_idx[:cfg[self.mode].RPN_POST_NMS_TOP_N]

        return cur_scores[keep_idx], cur_proposals[keep_idx]
