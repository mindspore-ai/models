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
"""TextFuseNet based on ResNet101."""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from .resnet101 import ResNetFea, ResidualBlockUsing
from .bbox_assign_sample_stage2 import BboxAssignSampleForRcnn
from .fpn_neck import FeatPyramidNeck
from .proposal_generator import Proposal
from .rcnn_cls import RcnnCls
from .rcnn_mask import RcnnMask
from .rpn import RPN
from .roi_align import SingleRoIExtractor
from .anchor_generator import AnchorGenerator
from .rcnn_seg import RcnnSeg
from .mutil_path_fuse import MutilPathFuseModule
class Text_Fuse_Net_Resnet101(nn.Cell):
    """
    TextFuseNet Network.

    Note:
        backbone = resnet101

    Returns:
        Tuple, tuple of output tensor.
        rpn_loss: Scalar, Total loss of RPN subnet.
        rcnn_loss: Scalar, Total loss of RCNN subnet.
        rpn_cls_loss: Scalar, Classification loss of RPN subnet.
        rpn_reg_loss: Scalar, Regression loss of RPN subnet.
        rcnn_cls_loss: Scalar, Classification loss of RCNNcls subnet.
        rcnn_reg_loss: Scalar, Regression loss of RCNNcls subnet.
        rcnn_mask_loss: Scalar, mask loss of RCNNmask subnet.

    Examples:
        net = Text_Fuse_Net_Resnet101()
    """
    def __init__(self, config):
        super(Text_Fuse_Net_Resnet101, self).__init__()
        self.config = config
        self.train_batch_size = config.batch_size
        self.test_roi_number = config.test_roi_number
        self.num_classes = config.num_classes
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_strides = config.anchor_strides
        self.target_means = tuple(config.rcnn_target_means)
        self.target_stds = tuple(config.rcnn_target_stds)
        self.image_height = config.img_height
        self.image_width = config.img_width
        self.init_anchor()

        # Backbone resnet101
        self.backbone = ResNetFea(ResidualBlockUsing,
                                  config.resnet_block,
                                  config.resnet_in_channels,
                                  config.resnet_out_channels,
                                  False)

        # Fpn
        self.fpn_ncek = FeatPyramidNeck(config.fpn_in_channels,
                                        config.fpn_out_channels,
                                        config.fpn_num_outs)

        # Rpn and rpn loss
        self.gt_labels_stage1 = Tensor(np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8))
        self.rpn_with_loss = RPN(config,
                                 self.train_batch_size,
                                 config.rpn_in_channels,
                                 config.rpn_feat_channels,
                                 config.num_anchors,
                                 config.rpn_cls_out_channels)

        # Proposal
        self.proposal_generator = Proposal(config,
                                           self.train_batch_size,
                                           config.activate_num_classes,
                                           config.use_sigmoid_cls)
        self.proposal_generator.set_train_local(config, True)
        self.proposal_generator_test = Proposal(config,
                                                config.test_batch_size,
                                                config.activate_num_classes,
                                                config.use_sigmoid_cls)
        self.proposal_generator_test.set_train_local(config, False)

        # Assign and sampler stage two
        self.bbox_assigner_sampler_for_rcnn = BboxAssignSampleForRcnn(config, self.train_batch_size,
                                                                      config.num_bboxes_stage2, True)
        self.decode = P.BoundingBoxDecode(max_shape=(768, 1280), means=self.target_means, \
                                          stds=self.target_stds)

        # Roi
        self.roi_align = SingleRoIExtractor(config,
                                            config.roi_layer,
                                            config.roi_align_out_channels,
                                            config.roi_align_featmap_strides,
                                            self.train_batch_size,
                                            config.roi_align_finest_scale,
                                            mask=False)
        self.roi_align.set_train_local(config, True)

        self.roi_align_mask = SingleRoIExtractor(config,
                                                 config.roi_layer,
                                                 config.roi_align_out_channels,
                                                 config.roi_align_featmap_strides,
                                                 self.train_batch_size,
                                                 config.roi_align_finest_scale,
                                                 mask=True)
        self.roi_align_mask.set_train_local(config, True)

        self.roi_align_test = SingleRoIExtractor(config,
                                                 config.roi_layer,
                                                 config.roi_align_out_channels,
                                                 config.roi_align_featmap_strides,
                                                 1,
                                                 config.roi_align_finest_scale,
                                                 mask=False)
        self.roi_align_test.set_train_local(config, False)

        self.roi_align_mask_test = SingleRoIExtractor(config,
                                                      config.roi_layer,
                                                      config.roi_align_out_channels,
                                                      config.roi_align_featmap_strides,
                                                      1,
                                                      config.roi_align_finest_scale,
                                                      mask=True)
        self.roi_align_mask_test.set_train_local(config, False)

        # Rcnn
        self.rcnn_cls = RcnnCls(config, self.train_batch_size, self.num_classes)
        self.rcnn_mask = RcnnMask(config, self.train_batch_size, self.num_classes)
        # Seg and mfa
        self.rcnn_seg = RcnnSeg(config, num_classes=1, image_shape=(self.image_height, self.image_width))
        self.fuse_module = MutilPathFuseModule(config)
        self.init_test()

    def init_anchor(self):
        """init anchor"""
        # Anchor generator
        anchor_base_sizes = None
        self.anchor_base_sizes = list(
            self.anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        featmap_sizes = self.config.feature_shapes
        assert len(featmap_sizes) == len(self.anchor_generators)

        self.anchor_list = self.get_anchors(featmap_sizes)

    def init_test(self):
        """init test"""
        # Op declare
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()
        self.topk = P.TopK()
        self.expand = P.ExpandDims()
        self.ga = P.GatherNd()

        self.concat = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)
        self.reshape = P.Reshape()
        self.select = P.Select()
        self.greater = P.Greater()
        self.transpose = P.Transpose()

        # Test mode
        self.test_batch_size = self.config.test_batch_size
        self.split = P.Split(axis=0, output_num=self.test_batch_size)
        self.split_shape = P.Split(axis=0, output_num=4)
        self.split_scores = P.Split(axis=1, output_num=self.num_classes)
        self.split_fb_mask = P.Split(axis=1, output_num=self.num_classes)
        self.split_cls = P.Split(axis=0, output_num=self.num_classes - 1)
        self.tile = P.Tile()
        self.gather = P.GatherNd()
        self.squeeze = P.Squeeze()

        self.rpn_max_num = self.config.rpn_max_num

        self.zeros_for_nms = Tensor(np.zeros((self.rpn_max_num, 3)).astype(np.float16))
        self.ones_mask = np.ones((self.rpn_max_num, 1)).astype(np.bool)
        self.zeros_mask = np.zeros((self.rpn_max_num, 1)).astype(np.bool)
        self.bbox_mask = Tensor(np.concatenate((self.ones_mask, self.zeros_mask,
                                                self.ones_mask, self.zeros_mask), axis=1))
        self.nms_pad_mask = Tensor(np.concatenate((self.ones_mask, self.ones_mask,
                                                   self.ones_mask, self.ones_mask, self.zeros_mask), axis=1))

        self.test_score_thresh = Tensor(np.ones((self.rpn_max_num, 1)).astype(np.float16) * self.config.test_score_thr)
        self.test_score_zeros = Tensor(np.ones((self.rpn_max_num, 1)).astype(np.float16) * 0)
        self.test_box_zeros = Tensor(np.ones((self.rpn_max_num, 4)).astype(np.float16) * -1)
        self.test_iou_thr = Tensor(np.ones((self.rpn_max_num, 1)).astype(np.float16) * self.config.test_iou_thr)
        self.test_max_per_img = self.config.test_max_per_img
        self.nms_test = P.NMSWithMask(self.config.test_iou_thr)
        self.softmax = P.Softmax(axis=1)
        self.logicand = P.LogicalAnd()
        self.oneslike = P.OnesLike()
        self.test_topk = P.TopK(sorted=True)
        self.test_num_proposal = self.test_batch_size * self.rpn_max_num

        # Improve speed
        self.concat_start = min(self.num_classes - 2, 55)
        self.concat_end = (self.num_classes - 1)

        # Init tensor
        roi_align_index = [np.array(np.ones((self.config.num_expected_pos_stage2 +
                                             self.config.num_expected_neg_stage2, 1)) * i,
                                    dtype=np.float16) for i in range(self.train_batch_size)]

        roi_align_index_test = [np.array(np.ones((self.config.rpn_max_num, 1)) * i, dtype=np.float16) \
                                for i in range(self.test_batch_size)]

        roi_align_mask_index_test = [np.array(np.ones((self.config.test_roi_number, 1)) * i, dtype=np.float16) for i in
                                     range(self.test_batch_size)]

        self.roi_align_index_tensor = Tensor(np.concatenate(roi_align_index))
        self.roi_align_index_test_tensor = Tensor(np.concatenate(roi_align_index_test))

        self.roi_align_mask_index_test_tensor = Tensor(np.concatenate(roi_align_mask_index_test))

        roi_align_index_pos = [np.array(np.ones((self.config.num_expected_pos_stage2, 1)) * i,
                                        dtype=np.float16) for i in range(self.train_batch_size)]
        self.roi_align_index_tensor_pos = Tensor(np.concatenate(roi_align_index_pos))

        self.rcnn_loss_cls_weight = Tensor(np.array(self.config.rcnn_loss_cls_weight).astype(np.float16))
        self.rcnn_loss_reg_weight = Tensor(np.array(self.config.rcnn_loss_reg_weight).astype(np.float16))
        self.rcnn_loss_mask_fb_weight = Tensor(np.array(self.config.rcnn_loss_mask_fb_weight).astype(np.float16))

        self.argmax_with_value = P.ArgMaxWithValue(axis=1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.onehot = P.OneHot()
        self.reducesum = P.ReduceSum()
        self.sigmoid = P.Sigmoid()
        self.expand_dims = P.ExpandDims()
        self.test_mask_fb_zeros = Tensor(np.zeros((self.rpn_max_num, 28, 28)).astype(np.float16))
        self.value = Tensor(1.0, mstype.float16)
        self.ones = Tensor(np.ones((self.config.test_roi_number, 1)).astype(np.int32))

    def label_and_sample_proposals(self, gt_bboxes, gt_labels, gt_valids, gt_masks, proposal_mask, proposal):
        """label and sample proposals"""
        bboxes_tuple = ()
        deltas_tuple = ()
        labels_tuple = ()
        mask_tuple = ()

        pos_bboxes_tuple = ()
        pos_mask_fb_tuple = ()
        pos_labels_tuple = ()
        pos_mask_tuple = ()

        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

                gt_masks_i = self.squeeze(gt_masks[i:i + 1:1, ::])
                gt_masks_i = self.cast(gt_masks_i, mstype.bool_)

                bboxes, deltas, labels, mask, pos_bboxes, pos_mask_fb, pos_labels, pos_mask = \
                    self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i,
                                                        gt_labels_i,
                                                        proposal_mask[i],
                                                        proposal[i][::, 0:4:1],
                                                        gt_valids_i,
                                                        gt_masks_i)
                bboxes_tuple += (bboxes,)
                deltas_tuple += (deltas,)
                labels_tuple += (labels,)
                mask_tuple += (mask,)

                pos_bboxes_tuple += (pos_bboxes,)
                pos_mask_fb_tuple += (pos_mask_fb,)
                pos_labels_tuple += (pos_labels,)
                pos_mask_tuple += (pos_mask,)

            bbox_targets = self.concat(deltas_tuple)
            rcnn_labels = self.concat(labels_tuple)
            bbox_targets = F.stop_gradient(bbox_targets)
            rcnn_labels = F.stop_gradient(rcnn_labels)
            rcnn_labels = self.cast(rcnn_labels, mstype.int32)

            rcnn_pos_masks_fb = self.concat(pos_mask_fb_tuple)
            rcnn_pos_masks_fb = F.stop_gradient(rcnn_pos_masks_fb)
            rcnn_pos_labels = self.concat(pos_labels_tuple)
            rcnn_pos_labels = F.stop_gradient(rcnn_pos_labels)
            rcnn_pos_labels = self.cast(rcnn_pos_labels, mstype.int32)
        else:
            mask_tuple += proposal_mask
            bbox_targets = proposal_mask
            rcnn_labels = proposal_mask

            rcnn_pos_masks_fb = proposal_mask
            rcnn_pos_labels = proposal_mask
            for p_i in proposal:
                bboxes_tuple += (p_i[::, 0:4:1],)
        return bboxes_tuple, pos_bboxes_tuple, mask_tuple, pos_mask_tuple, bbox_targets, \
               rcnn_labels, rcnn_pos_labels, rcnn_pos_masks_fb

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids, gt_masks):
        """textfusenet forward"""
        x = self.backbone(img_data)
        x = self.fpn_ncek(x)

        rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss, _ = self.rpn_with_loss(x,
                                                                                           img_metas,
                                                                                           self.anchor_list,
                                                                                           gt_bboxes,
                                                                                           self.gt_labels_stage1,
                                                                                           gt_valids)

        if self.training:
            proposal, proposal_mask = self.proposal_generator(cls_score, bbox_pred, self.anchor_list)
        else:
            proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred, self.anchor_list)
        if gt_labels is not None:
            gt_labels = self.cast(gt_labels, mstype.int32)
            gt_valids = self.cast(gt_valids, mstype.int32)
        bboxes_tuple, pos_bboxes_tuple, mask_tuple, pos_mask_tuple, bbox_targets, \
        rcnn_labels, rcnn_pos_labels, rcnn_pos_masks_fb = self.label_and_sample_proposals(gt_bboxes,
                                                                                          gt_labels,
                                                                                          gt_valids,
                                                                                          gt_masks,
                                                                                          proposal_mask,
                                                                                          proposal)
        pos_rois = None
        if self.training:
            if self.train_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
                pos_bboxes_all = self.concat(pos_bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
                pos_bboxes_all = pos_bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_tensor, bboxes_all))
            pos_rois = self.concat_1((self.roi_align_index_tensor_pos, pos_bboxes_all))
            pos_rois = self.cast(pos_rois, mstype.float32)
            pos_rois = F.stop_gradient(pos_rois)
        else:
            if self.test_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))

        rois = self.cast(rois, mstype.float32)
        rois = F.stop_gradient(rois)

        if self.training:
            roi_feats = self.roi_align(rois,
                                       self.cast(x[0], mstype.float32),
                                       self.cast(x[1], mstype.float32),
                                       self.cast(x[2], mstype.float32),
                                       self.cast(x[3], mstype.float32))
        else:
            roi_feats = self.roi_align_test(rois,
                                            self.cast(x[0], mstype.float32),
                                            self.cast(x[1], mstype.float32),
                                            self.cast(x[2], mstype.float32),
                                            self.cast(x[3], mstype.float32))

        roi_feats = self.cast(roi_feats, mstype.float16)
        rcnn_masks = self.concat(mask_tuple)
        rcnn_masks = F.stop_gradient(rcnn_masks)
        rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))

        rcnn_pos_masks = self.concat(pos_mask_tuple)
        rcnn_pos_masks = F.stop_gradient(rcnn_pos_masks)
        rcnn_pos_mask_squeeze = self.squeeze(self.cast(rcnn_pos_masks, mstype.bool_))
        rcnn_cls_loss, rcnn_reg_loss = self.rcnn_cls(roi_feats,
                                                     bbox_targets,
                                                     rcnn_labels,
                                                     rcnn_mask_squeeze)

        output = ()
        if self.training:
            roi_feats_mask = self.roi_align_mask(pos_rois,
                                                 self.cast(x[0], mstype.float32),
                                                 self.cast(x[1], mstype.float32),
                                                 self.cast(x[2], mstype.float32),
                                                 self.cast(x[3], mstype.float32))
            seg_loss, global_context = self.rcnn_seg(x[:4], gt_masks, gt_labels, pos_rois)
            roi_feats_mask = self.cast(roi_feats_mask, mstype.float16)

            roi_feats_mask = self.fuse_module(roi_feats_mask, rcnn_pos_labels, global_context, pos_rois)

            roi_feats_mask = self.cast(roi_feats_mask, mstype.float16)
            rcnn_mask_fb_loss = self.rcnn_mask(roi_feats_mask,
                                               rcnn_pos_labels,
                                               rcnn_pos_mask_squeeze,
                                               rcnn_pos_masks_fb)

            rcnn_loss = self.rcnn_loss_cls_weight * rcnn_cls_loss + self.rcnn_loss_reg_weight * rcnn_reg_loss + \
                        self.rcnn_loss_mask_fb_weight * rcnn_mask_fb_loss

            rcnn_loss += seg_loss

            output += (rpn_loss, rcnn_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss,
                       rcnn_mask_fb_loss, seg_loss)
        else:
            out = self.get_det_bboxes(rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, bboxes_all)
            output = self.get_result(out, x)

        return output

    def get_result(self, out, feat):
        """Get the actual result"""
        bboxes = out[0]
        labels = out[1]
        masks = out[2]
        _, index = self.topk(self.cast(self.squeeze(masks), mstype.float32), self.test_roi_number)
        index = self.expand(index, 1)

        post_boxes = self.ga(self.squeeze(bboxes), index)
        post_labels = self.ga(self.expand(self.squeeze(labels), 1), index)
        post_masks = self.ga(self.expand(self.squeeze(masks), 1), index)

        rois = self.concat_1((self.roi_align_mask_index_test_tensor, post_boxes[:, :4]))
        _, global_context = self.rcnn_seg(feat[:4], None, None, rois)
        mask_fb_pred_all = self.roi_align_mask_test(rois, self.cast(feat[0], mstype.float32),
                                                    self.cast(feat[1], mstype.float32),
                                                    self.cast(feat[2], mstype.float32),
                                                    self.cast(feat[3], mstype.float32))
        mask_fb_pred_all = self.fuse_module(mask_fb_pred_all, post_labels[:, 0], global_context, rois)
        mask_fb_pred_all = self.rcnn_mask(mask_fb_pred_all)
        mask_fb_pred_all = self.sigmoid(mask_fb_pred_all)
        mask_fb_pred_all = self.expand(mask_fb_pred_all, 0)
        post_boxes = self.expand(post_boxes, 0)
        post_labels = self.expand(post_labels, 0)
        post_masks = self.expand(post_masks, 0)
        return post_boxes, post_labels, post_masks, mask_fb_pred_all

    def get_det_bboxes(self, cls_logits, reg_logits, mask_logits, rois):
        """Get the actual detection box."""
        scores = self.softmax(cls_logits / self.value)

        boxes_all = ()
        for i in range(self.num_classes):
            k = i * 4
            reg_logits_i = self.squeeze(reg_logits[::, k:k+4:1])
            out_boxes_i = self.decode(rois, reg_logits_i)
            boxes_all += (out_boxes_i,)

        scores_all = self.split(scores)
        mask_all = self.split(self.cast(mask_logits, mstype.int32))

        boxes_all_with_batchsize = ()
        for i in range(self.test_batch_size):
            boxes_tuple = ()
            for j in range(self.num_classes):
                boxes_tmp = self.split(boxes_all[j])
                out_boxes_h = boxes_tmp[i]
                out_boxes_w = boxes_tmp[i]
                boxes_tuple += (self.select(self.bbox_mask, out_boxes_w, out_boxes_h),)
            boxes_all_with_batchsize += (boxes_tuple,)

        output = self.multiclass_nms(boxes_all_with_batchsize, scores_all, mask_all)

        return output

    def multiclass_nms(self, boxes_all, scores_all, mask_all):
        """Multiscale postprocessing."""
        all_bboxes = ()
        all_labels = ()
        all_masks = ()

        for i in range(self.test_batch_size):
            bboxes = boxes_all[i]
            scores = scores_all[i]
            masks = self.cast(mask_all[i], mstype.bool_)

            res_boxes_tuple = ()
            res_labels_tuple = ()
            res_masks_tuple = ()

            for j in range(self.num_classes - 1):
                k = j + 1
                _cls_scores = scores[::, k:k + 1:1]
                _bboxes = self.squeeze(bboxes[k])
                _mask_o = self.reshape(masks, (self.rpn_max_num, 1))


                cls_mask = self.greater(_cls_scores, self.test_score_thresh)
                _mask = self.logicand(_mask_o, cls_mask)

                _reg_mask = self.cast(self.tile(self.cast(_mask, mstype.int32), (1, 4)), mstype.bool_)

                _bboxes = self.select(_reg_mask, _bboxes, self.test_box_zeros)
                _cls_scores = self.select(_mask, _cls_scores, self.test_score_zeros)
                __cls_scores = self.squeeze(_cls_scores)
                scores_sorted, topk_inds = self.test_topk(__cls_scores, self.rpn_max_num)
                topk_inds = self.reshape(topk_inds, (self.rpn_max_num, 1))
                scores_sorted = self.reshape(scores_sorted, (self.rpn_max_num, 1))
                _bboxes_sorted = self.gather(_bboxes, topk_inds)
                _mask_sorted = self.gather(_mask, topk_inds)

                scores_sorted = self.tile(scores_sorted, (1, 4))
                cls_dets = self.concat_1((_bboxes_sorted, scores_sorted))
                cls_dets = P.Slice()(cls_dets, (0, 0), (self.rpn_max_num, 5))

                cls_dets, _index, _mask_nms = self.nms_test(cls_dets)
                _index = self.reshape(_index, (self.rpn_max_num, 1))
                _mask_nms = self.reshape(_mask_nms, (self.rpn_max_num, 1))

                _mask_n = self.gather(_mask_sorted, _index)
                _mask_n = self.logicand(_mask_n, _mask_nms)

                cls_labels = self.oneslike(_index) * j
                res_boxes_tuple += (cls_dets,)
                res_labels_tuple += (cls_labels,)
                res_masks_tuple += (_mask_n,)

            res_boxes_start = self.concat(res_boxes_tuple[:self.concat_start])
            res_labels_start = self.concat(res_labels_tuple[:self.concat_start])
            res_masks_start = self.concat(res_masks_tuple[:self.concat_start])

            res_boxes_end = self.concat(res_boxes_tuple[self.concat_start:self.concat_end])
            res_labels_end = self.concat(res_labels_tuple[self.concat_start:self.concat_end])
            res_masks_end = self.concat(res_masks_tuple[self.concat_start:self.concat_end])

            res_boxes = self.concat((res_boxes_start, res_boxes_end))
            res_labels = self.concat((res_labels_start, res_labels_end))
            res_masks = self.concat((res_masks_start, res_masks_end))

            reshape_size = (self.num_classes - 1) * self.rpn_max_num
            res_boxes = self.reshape(res_boxes, (1, reshape_size, 5))
            res_labels = self.reshape(res_labels, (1, reshape_size, 1))
            res_masks = self.reshape(res_masks, (1, reshape_size, 1))

            all_bboxes += (res_boxes,)
            all_labels += (res_labels,)
            all_masks += (res_masks,)

        all_bboxes = self.concat(all_bboxes)
        all_labels = self.concat(all_labels)
        all_masks = self.concat(all_masks)

        return all_bboxes, all_labels, all_masks

    def get_anchors(self, featmap_sizes):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = ()
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors += (Tensor(anchors.astype(np.float16)),)

        return multi_level_anchors

    def rcnn_mask_test(self, x, rois, cls_pred, reg_pred):
        """Prediction masks in an images by the bounding boxes
        """
        cls_scores = self.softmax(cls_pred / self.value)

        cls_scores_all = self.split(cls_scores)
        reg_pred = self.reshape(reg_pred, (-1, self.num_classes, 4))
        reg_pred_all = self.split(reg_pred)
        rois_all = self.split(rois)
        boxes_tuple = ()
        for i in range(self.test_batch_size):
            cls_score_max_index, _ = self.argmax_with_value(cls_scores_all[i])
            cls_score_max_index = self.cast(self.onehot(cls_score_max_index, self.num_classes,
                                                        self.on_value, self.off_value), mstype.float16)
            cls_score_max_index = self.expand_dims(cls_score_max_index, -1)
            cls_score_max_index = self.tile(cls_score_max_index, (1, 1, 4))
            reg_pred_max = reg_pred_all[i] * cls_score_max_index
            reg_pred_max = self.reducesum(reg_pred_max, 1)
            out_boxes_i = self.decode(rois_all[i], reg_pred_max)
            boxes_tuple += (out_boxes_i,)

        boxes_all = self.concat(boxes_tuple)
        boxes_rois = self.concat_1((self.roi_align_index_test_tensor, boxes_all))
        boxes_rois = self.cast(boxes_rois, mstype.float32)
        roi_feats_mask_test = self.roi_align_mask_test(boxes_rois,
                                                       self.cast(x[0], mstype.float32),
                                                       self.cast(x[1], mstype.float32),
                                                       self.cast(x[2], mstype.float32),
                                                       self.cast(x[3], mstype.float32))
        return roi_feats_mask_test, boxes_rois


class TextFuseNet_Infer(nn.Cell):
    def __init__(self, config):
        super(TextFuseNet_Infer, self).__init__()
        self.network = Text_Fuse_Net_Resnet101(config)
        self.network.set_train(False)

    def construct(self, img_data, img_metas):
        output = self.network(img_data, img_metas, None, None, None, None)
        return output
