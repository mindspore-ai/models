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
'''maskrcnnPointRend R50'''

import numpy
import mindspore
import mindspore.nn as nn
from mindspore.ops import constexpr
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import functional as F
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore.numpy as np
from maskrcnn_pointrend.src.point_rend.sampling_points import get_int
from maskrcnn_pointrend.src.point_rend.coarse_mask_head import CoarseMaskHead
from maskrcnn_pointrend.src.point_rend.sampling_points import generate_regular_grid_point_coords
from maskrcnn_pointrend.src.point_rend.sampling_points import point_sample_fine_grained_features
from maskrcnn_pointrend.src.point_rend.sampling_points import point_sample
from maskrcnn_pointrend.src.point_rend.point_head import StandardPointHead
from maskrcnn_pointrend.src.point_rend.sampling_points import batch_nms
from maskrcnn.mask_rcnn_mobilenetv1 import Mask_Rcnn_Mobilenetv1
from maskrcnn.maskrcnn_mobilenetv1.roi_align import ROIAlign

class maskrcnn_r50_pointrend(Mask_Rcnn_Mobilenetv1):
    '''maskrcnn_r50_pointrend'''
    def __init__(self, config):
        super(maskrcnn_r50_pointrend, self).__init__(config)
        self.mask_coarse_side_size = 14
        self.mask_coarse_in_features = ("0",)
        self.mask_coarse_in_features_int = (0,)
        self._feature_scales = {"0": 0.25, "1": 0.125, "2": 0.0625, "3": 0.03125, "4": 0.015625}
        self.not_mask = config.not_mask
        if self.not_mask:
            num_classes = self.num_classes - 1
        else:
            num_classes = self.num_classes
        self.mask_coarse_head = CoarseMaskHead(num_classes=num_classes)
        self.mask_point_head = StandardPointHead(input_channels=256, num_classes=num_classes)
        self.mask_point_in_features_int = (0,)
        self.mask_point_in_features = ("0",)
        self.mask_point_train_num_points = 14 * 14
        self.mask_point_oversample_ratio = 3
        self.mask_point_importance_sample_ratio = 0.75

        self.mask_point_subdivision_steps = 3
        self.mask_point_subdivision_num_points = 28 * 28
        self.rcnn_loss_mask_coarse_weight = Tensor(config.rcnn_loss_mask_coarse_weight, mindspore.float32)
        self.rcnn_loss_mask_point_weight = Tensor(config.rcnn_loss_mask_point_weight, mindspore.float32)
        self.img_shape = [768, 1280]
        self.test_score_thresh = 0.05
        self.test_nms_thresh = 0.5
        self.test_topk_per_image = 100
        self.weights = [10.0, 10.0, 5.0, 5.0]
        self.mask_point_subdivision_init_resolution = 28
        self.mask_point_subdivision_num_points = 28 * 28
        self.scatter_ = ops.TensorScatterUpdate()
        self.mean_loss = P.ReduceMean()
        self.maximum = P.Maximum()
        self.sum_loss = P.ReduceSum()
        self.op_expanddims = ops.ExpandDims()
        self.op_roi_align = ROIAlign(7, 7, 1.0, sample_num=2, roi_align_mode=0)
        self.op_squeeze = ops.Squeeze(1)
        self.op_tile = ops.Tile()
        self.bce_loss = nn.BCEWithLogitsLoss('none')
        self.uniformreal = ops.UniformReal(seed=2)
        self.topk = ops.TopK(sorted=True)
        self.Print = ops.Print()


    def roi_mask_point_loss(self, mask_logits, points_coord, fg_labels, fg_masks):
        '''roi_mask_point_loss'''
        num_boxes = len(fg_masks)
        point_coords_splits = np.split(points_coord, num_boxes)
        gt_mask_logits = []
        for i, gt_mask in enumerate(fg_masks):
            h, w = gt_mask.shape[1:]
            scale = get_tensor([w, h], mindspore.float32)
            points_coord_grid_sample_format = point_coords_splits[i] / scale
            gt_mask = self.op_expanddims(gt_mask, 1)
            gt_mask_logit = point_sample(gt_mask, points_coord_grid_sample_format)
            gt_mask_logits.append(gt_mask_logit)
        concat = P.Concat(axis=0)
        gt_mask_logits = concat(gt_mask_logits)
        fg_labels = concat(fg_labels)
        weights = fg_labels != 0
        weights = weights.astype(mindspore.float32)
        if gt_mask_logits.shape[0] == 0:
            return mask_logits.sum() * 0
        total_num_masks = mask_logits.shape[0]
        indices = np.arange(total_num_masks)
        mask_logits = mask_logits[indices, fg_labels]
        squeeze = ops.Squeeze(1)
        gt_mask_logits = squeeze(gt_mask_logits)
        point_loss = self.bce_loss(mask_logits, gt_mask_logits)
        point_loss = self.mean_loss(point_loss, 1)
        point_loss = point_loss * weights
        if self.platform == "CPU" or self.platform == "GPU":
            sum_weight = self.sum_loss(weights, (0,))
            point_loss = point_loss / self.maximum(self.op_expanddims(sum_weight, 0), 1)
        else:
            point_loss = point_loss / self.sum_loss(weights, (0,))
        point_loss = self.sum_loss(point_loss)
        return point_loss

    def project_masks_on_boxes(self, gt_masks, rois):
        '''project_masks_on_boxes'''
        gt_masks = self.op_expanddims(gt_masks, 1)
        gt_masks = self.cast(gt_masks, mstype.float32)
        batch_inds = np.arange(len(gt_masks))[:, None]
        rois = self.concat_1((batch_inds.astype(mstype.float32), rois))
        output = self.op_roi_align(gt_masks, rois)
        output = output >= 0.5
        return output

    def maskrcnn_loss(self, mask_logits, fg_proposals, gt_labels, pos_masks, fg_masks):
        '''maskrcnn_loss'''
        gt_masks = []
        for i, fg_mask in enumerate(fg_masks):
            mask_target = self.project_masks_on_boxes(fg_mask, fg_proposals[i])
            gt_masks.append(mask_target)
        gt_masks = self.concat(gt_masks)
        gt_labels = self.concat(gt_labels)
        pos_masks = self.cast(self.concat(pos_masks), mindspore.float32)
        indices = np.arange(gt_labels.shape[0])
        mask_logits = mask_logits[indices, gt_labels]
        gt_masks = self.op_squeeze(gt_masks)
        gt_masks = self.cast(gt_masks, mstype.float32)
        loss_mask_coarse = self.bce_loss(mask_logits, gt_masks)
        loss_mask_coarse = self.mean_loss(loss_mask_coarse, (1, 2))
        loss_mask_coarse = loss_mask_coarse * pos_masks
        if self.platform == "CPU" or self.platform == "GPU":
            sum_weight = self.sum_loss(pos_masks, (0,))
            loss_mask_coarse = loss_mask_coarse / self.maximum(self.op_expanddims(sum_weight, 0), 1)
        else:
            loss_mask_coarse = loss_mask_coarse / self.sum_loss(pos_masks, (0,))
        loss_mask_coarse = self.sum_loss(loss_mask_coarse)
        return loss_mask_coarse

    def _forward_mask_coarse(self, features, boxes):
        '''_forward_mask_coarse'''
        pos_proposal = self.concat(boxes)
        point_coords = generate_regular_grid_point_coords(
            pos_proposal.shape[0], self.mask_coarse_side_size
        )
        mask_coarse_features_list = []
        features_scales = []
        for k in self.mask_coarse_in_features_int:
            mask_coarse_features_list.append(features[k])
        for k in self.mask_coarse_in_features:
            features_scales.append(self._feature_scales[k])
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list, features_scales, boxes, point_coords
        )
        mask_coarse_logits = self.mask_coarse_head(mask_features)
        return mask_coarse_logits

    def _forward_mask_point(self, features, mask_coarse_logits, fg_proposals, fg_labels, fg_masks):
        '''_forward_mask_point'''
        mask_features_list = []
        features_scales = []
        for k in self.mask_point_in_features_int:
            mask_features_list.append(features[k])
        for k in self.mask_point_in_features:
            features_scales.append(self._feature_scales[k])
        gt_classes = self.concat(fg_labels)
        point_coords = self.SamplingPoints(mask_coarse_logits,
                                           self.mask_point_train_num_points,
                                           self.mask_point_oversample_ratio,
                                           self.mask_point_importance_sample_ratio,
                                           gt_classes)
        fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
            mask_features_list, features_scales, fg_proposals, point_coords
        )
        coarse_features = point_sample(mask_coarse_logits, point_coords)
        point_logits = self.mask_point_head(fine_grained_features,
                                            coarse_features)
        if self.not_mask:
            return self.roi_mask_point_loss2(
                point_logits, point_coords_wrt_image, fg_labels, fg_masks
            )
        return self.roi_mask_point_loss(
            point_logits, point_coords_wrt_image, fg_labels, fg_masks
        )

    def _subdivision_inference(self, features, coarse_mask, pred_boxes, pred_classes):
        '''_subdivision_inference'''
        mask_logits = None
        pred_classes = self.concat(pred_classes)
        resize_bilinear = nn.ResizeBilinear()
        mask_features_list = []
        features_scales = []
        for k in self.mask_point_in_features_int:
            mask_features_list.append(features[k])
        for k in self.mask_point_in_features:
            features_scales.append(self._feature_scales[k])
        points_idx = None
        for _ in range(self.mask_point_subdivision_steps + 1):
            if mask_logits is None:
                point_coords = generate_regular_grid_point_coords(
                    pred_classes.shape[0],
                    self.mask_point_subdivision_init_resolution,
                )
            else:
                mask_logits = resize_bilinear(mask_logits, scale_factor=2, align_corners=False)
                gt_class_logits = mask_logits[
                    np.arange(mask_logits.shape[0]), pred_classes
                ]
                gt_class_logits = self.expand_dims(gt_class_logits, 1)
                gt_class_logits = -gt_class_logits.abs()
                points_idx, point_coords = self.SamplingPoints(gt_class_logits,
                                                               self.mask_point_subdivision_num_points,
                                                               training=self.training)
            fine_grained_features, _ = point_sample_fine_grained_features(
                mask_features_list, features_scales, pred_boxes, point_coords
            )
            coarse_features = point_sample(coarse_mask, point_coords)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)

            if mask_logits is None:
                R, C, _ = point_logits.shape
                mask_logits = point_logits.reshape(
                    R,
                    C,
                    self.mask_point_subdivision_init_resolution,
                    self.mask_point_subdivision_init_resolution,
                )
                if pred_classes.shape[0] == 0:
                    return self.mask_rcnn_inference(mask_logits, pred_boxes, pred_classes)
            else:
                B, C, H, W = mask_logits.shape
                points_idx = self.tile(points_idx[:, None], (1, C, 1))
                _, _, x = points_idx.shape
                index_B = []
                index_C = []
                for b in range(B):
                    index_B += [b] * C * x
                    for c in range(C):
                        index_C += [c] * x
                index_B = Tensor(index_B, mindspore.int32)
                index_C = Tensor(index_C, mindspore.int32)
                index = self.concat_1((index_B[:, None], index_C[:, None], points_idx.view(-1)[:, None]))
                point_logits = point_logits.view(-1)
                mask_logits = mask_logits.view(B, C, -1)
                mask_logits = self.scatter_(mask_logits, index, point_logits).view(B, C, H, W)
        return self.mask_rcnn_inference(mask_logits, pred_boxes, pred_classes)

    def mask_rcnn_inference(self, pred_mask_logits, pre_boxes, pre_class):
        '''mask_rcnn_inference'''
        cls_agnostic_mask = pred_mask_logits.shape[1] == 1
        sigmoid = nn.Sigmoid()
        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            num_masks = pred_mask_logits.shape[0]
            class_pred = pre_class
            indices = np.arange(num_masks)
            mask_probs_pred = sigmoid(pred_mask_logits[indices, class_pred][:, None])
        num_boxes = len(pre_boxes)
        mask_probs_pred = np.split(mask_probs_pred, num_boxes)
        pred_masks = []
        for prob in mask_probs_pred:
            pred_masks.append(prob)
        return pred_masks

    def predict_boxes(self, proposal_deltas, proposals):
        '''predict_boxes'''
        concat = ops.Concat(0)
        proposal_boxes = concat(proposals)
        boxes_all = ()
        for i in range(self.num_classes):
            k = i * 4
            reg_logits_i = self.squeeze(proposal_deltas[::, k:k + 4:1])
            out_boxes_i = self.decode(proposal_boxes, reg_logits_i)
            boxes_all += (out_boxes_i,)
        return self.split(self.concat_1(boxes_all))

    def predict_probs(self, scores):
        '''predict_probs'''
        softmax = nn.Softmax(-1)
        probs = softmax(scores)
        return self.split(probs)

    def fast_rcnn_inference(self, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image):
        '''fast_rcnn_inference'''
        box_per_image = []
        score_per_image = []
        id2class_per_image = []
        for scores_per_image, boxes_per_image in zip(scores, boxes):
            result = self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            if result:
                box_per_image.append(result[0])
                score_per_image.append(result[1])
                id2class_per_image.append(result[2])
        return box_per_image, score_per_image, id2class_per_image

    def fast_rcnn_inference_single_image(self, boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image):
        '''fast_rcnn_inference_single_image'''
        isfinite = ops.IsFinite()
        logicaland = ops.LogicalAnd()
        valid_mask = logicaland(isfinite(boxes).all(1), isfinite(scores).all(1))
        if not valid_mask.all():
            valid_mask = valid_mask.astype(mindspore.int32)
            tile = ops.Tile()
            boxes = boxes * tile(valid_mask[:, None], (1, boxes.shape[1]))
            scores = scores * tile(valid_mask[:, None], (1, scores.shape[1]))
        scores = scores[:, 1:]
        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = boxes.reshape(-1, 4)
        h, w = image_shape
        x1 = boxes[:, 0].clip(xmin=0, xmax=w)
        y1 = boxes[:, 1].clip(xmin=0, xmax=h)
        x2 = boxes[:, 2].clip(xmin=0, xmax=w)
        y2 = boxes[:, 3].clip(xmin=0, xmax=h)
        stack = ops.Stack(-1)
        boxes = stack((x1, y1, x2, y2))
        boxes = boxes.view(-1, num_bbox_reg_classes, 4)
        boxes = boxes[:, 1:, :]
        filter_mask = scores > score_thresh
        if filter_mask.astype(mindspore.int32).sum() == 0:
            return []
        data = numpy.nonzero(filter_mask.asnumpy())
        filter_inds = get_tensor(data, mindspore.int32)
        filter_inds = filter_inds.T
        gathernd = ops.GatherNd()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = gathernd(boxes, filter_inds)
        scores = gathernd(scores, filter_inds)
        boxes = boxes.asnumpy()
        scores = scores.asnumpy()
        filter_inds = filter_inds.asnumpy()
        keep = batch_nms(boxes, filter_inds[:, 1], scores, nms_thresh)
        if not keep:
            return []
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        boxes = Tensor(boxes, mindspore.float32)
        scores = Tensor(scores, mindspore.float32)
        filter_inds = Tensor(filter_inds, mindspore.int32)
        return boxes, scores, filter_inds

    def SamplingPoints(self, mask, N, k=3, beta=0.75, gt_classes=None, training=True):
        '''SamplingPoints'''
        B, _, H, W = mask.shape
        if not training:
            H_step, W_step = 1 / H, 1 / W
            N_ = H * W
            if N_ < N:
                N = N_
            uncertainty_map = mask
            topk = ops.TopK(sorted=True)
            _, idx = topk(uncertainty_map.view(B, -1).astype(mindspore.float32), N)

            zeros = ops.Zeros()
            points = zeros((B, N, 2), mindspore.float32)
            points[:, :, 0] = W_step / 2.0 + (idx % W).astype(mindspore.float32) * W_step
            points[:, :, 1] = H_step / 2.0 + (idx // W).astype(mindspore.float32) * H_step
            return idx, points

        over_generation = self.uniformreal((B, k * N, 2))
        point_logits = point_sample(mask, over_generation)
        point_uncertainties = point_logits[
            np.arange(point_logits.shape[0]), gt_classes
        ]
        point_uncertainties = self.expand_dims(point_uncertainties, 1)
        point_uncertainties = -point_uncertainties.abs()
        _, idx = self.topk(point_uncertainties[:, 0, :].astype(mindspore.float32), get_int(beta * N))
        net = nn.Range(0, B)
        range_ = net()
        shift = (k * N) * (range_.astype(mindspore.int64))
        idx += shift[:, None]
        importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, get_int(beta * N), 2)
        coverage = self.uniformreal((B, N - get_int(beta * N), 2))
        return self.concat_1((importance, coverage))

    def pointrend(self, x, pos_labels_tuple, pos_mask_tuple, pos_bboxes_tuple, roi_pos_masks_tuple):
        '''pointrend'''
        if self.not_mask:
            return self.pointrend2(x, pos_labels_tuple, pos_mask_tuple, pos_bboxes_tuple, roi_pos_masks_tuple)
        fg_labels = []
        pos_masks = []
        for i, label in enumerate(pos_labels_tuple):
            pos_mask = pos_mask_tuple[i][:, 0].astype(mindspore.bool_)
            pos_mask = F.stop_gradient(pos_mask)
            fg_label = self.cast(self.logicand(self.greater(label, 0), pos_mask),
                                 mindspore.int32) * label
            fg_label = F.stop_gradient(fg_label)
            fg_labels.append(fg_label)
            pos_masks.append(pos_mask)
        mask_coarse_logits = self._forward_mask_coarse(x, pos_bboxes_tuple)  # (128,81,7,7)
        loss_mask_point = self._forward_mask_point(x, mask_coarse_logits, pos_bboxes_tuple, fg_labels,
                                                   roi_pos_masks_tuple)
        rcnn_mask_coarse_loss = self.maskrcnn_loss(mask_coarse_logits, pos_bboxes_tuple, fg_labels, pos_masks,
                                                   roi_pos_masks_tuple)
        return loss_mask_point, rcnn_mask_coarse_loss

    def get_output_eval(self, x, bboxes_all, rcnn_cls_loss, rcnn_reg_loss):
        '''get_output_eval'''
        boxes = self.predict_boxes(rcnn_reg_loss, bboxes_all)
        scores = self.predict_probs(rcnn_cls_loss)
        image_shapes = self.img_shape
        pred_boxes, pred_scores, id2class = self.fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
        if not pred_boxes:
            return []
        pred_classes = []
        if self.not_mask:
            for item in id2class:
                pred_classes.append(item[:, 1])
        else:
            for item in id2class:
                pred_classes.append(item[:, 1] + 1)
        mask_coarse_logits = self._forward_mask_coarse(x, pred_boxes)
        pred_masks = self._subdivision_inference(x, mask_coarse_logits, pred_boxes, pred_classes)
        if not self.not_mask:
            pred_classes = []
            for item in id2class:
                pred_classes.append(item[:, 1])
        return pred_boxes, pred_classes, pred_scores, pred_masks

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids, gt_masks):
        '''construct'''
        x = self.backbone(img_data)
        x = self.fpn_ncek(x)
        gt_valids = self.cast(gt_valids, mstype.bool_)
        _, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss, _ = self.rpn_with_loss(x,
                                                                                    img_metas,
                                                                                    self.anchor_list,
                                                                                    gt_bboxes,
                                                                                    self.gt_labels_stage1,
                                                                                    gt_valids)
        if self.training:
            proposal, proposal_mask = self.proposal_generator(cls_score, bbox_pred, self.anchor_list)

        else:
            proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred,
                                                                   self.anchor_list)
        gt_labels = self.cast(gt_labels, mstype.int32)
        gt_valids = self.cast(gt_valids, mstype.int32)
        bboxes_tuple = ()
        deltas_tuple = ()
        labels_tuple = ()
        mask_tuple = ()
        pos_bboxes_tuple = ()
        pos_mask_fb_tuple = ()
        pos_labels_tuple = ()
        pos_mask_tuple = ()
        gt_mask_tuple = ()
        roi_pos_masks_tuple = ()
        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)
                gt_masks_i = self.squeeze(gt_masks[i:i + 1:1, ::])
                gt_masks_i = self.cast(gt_masks_i, mstype.bool_)
                bboxes, deltas, labels, mask, pos_bboxes, pos_mask_fb, pos_labels, pos_mask, roi_pos_masks_fb = \
                    self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i, gt_labels_i, proposal_mask[i],
                                                        proposal[i][::, 0:4:1], gt_valids_i, gt_masks_i)
                bboxes_tuple += (bboxes,)
                deltas_tuple += (deltas,)
                labels_tuple += (labels,)
                mask_tuple += (mask,)
                pos_bboxes = F.stop_gradient(pos_bboxes)
                pos_bboxes_tuple += (pos_bboxes,)
                pos_mask_fb_tuple += (pos_mask_fb,)
                pos_labels_tuple += (pos_labels,)
                pos_mask_tuple += (pos_mask,)
                gt_mask_tuple += (gt_masks_i,)
                roi_pos_masks_tuple += (roi_pos_masks_fb,)
            bbox_targets = self.concat(deltas_tuple)
            rcnn_labels = self.concat(labels_tuple)
            bbox_targets = F.stop_gradient(bbox_targets)
            rcnn_labels = F.stop_gradient(rcnn_labels)
            rcnn_labels = self.cast(rcnn_labels, mstype.int32)
        else:
            mask_tuple += proposal_mask
            bbox_targets = proposal_mask
            rcnn_labels = proposal_mask
            for p_i in proposal:
                bboxes_tuple += (p_i[::, 0:4:1],)
        if self.training:
            if self.train_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_tensor, bboxes_all))
        else:
            if self.test_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))
        rois = F.stop_gradient(self.cast(rois, mstype.float32))
        if self.training:
            rcnn_cls_loss = rcnn_reg_loss = get_tensor(0, mindspore.float32)
        else:
            roi_feats = self.roi_align_test(rois, self.cast(x[0], mstype.float32),
                                            self.cast(x[1], mstype.float32),
                                            self.cast(x[2], mstype.float32),
                                            self.cast(x[3], mstype.float32))
            roi_feats = self.cast(roi_feats, mstype.float32)
            rcnn_masks = self.concat(mask_tuple)
            rcnn_masks = F.stop_gradient(rcnn_masks)
            rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))
            rcnn_cls_loss, rcnn_reg_loss = self.rcnn_cls(roi_feats, bbox_targets,
                                                         rcnn_labels, rcnn_mask_squeeze)
        output = ()
        if self.training:
            loss_mask_point, rcnn_mask_coarse_loss = self.pointrend(x, pos_labels_tuple, pos_mask_tuple,
                                                                    pos_bboxes_tuple, roi_pos_masks_tuple)
            output += (
                rcnn_mask_coarse_loss, loss_mask_point, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss,
                loss_mask_point)
            return output
        return self.get_output_eval(x, bboxes_tuple, rcnn_cls_loss, rcnn_reg_loss)

    def pointrend2(self, x, pos_labels_tuple, pos_mask_tuple, pos_bboxes_tuple, roi_pos_masks_tuple):
        '''pointrend2'''
        fg_labels = []
        fg_boxes = []
        fg_roi_masks = []
        for i, label in enumerate(pos_labels_tuple):
            pos_mask = pos_mask_tuple[i][:, 0].astype(mindspore.bool_)
            pos_mask = F.stop_gradient(pos_mask)
            pos_box = pos_bboxes_tuple[i]
            roi_mask = roi_pos_masks_tuple[i]
            fg_label = self.cast(self.logicand(self.greater(label, 0), pos_mask),
                                 mindspore.int32) * label
            fg_idxs = fg_label.nonzero().squeeze(1)
            fg_label = fg_label[fg_idxs]
            pos_box = pos_box[fg_idxs]
            fg_label = fg_label-1
            fg_label = F.stop_gradient(fg_label)
            fg_labels.append(fg_label)
            fg_roi_masks.append(roi_mask)
            fg_boxes.append(pos_box)
        mask_coarse_logits = self._forward_mask_coarse(x, fg_boxes)
        loss_mask_point = self._forward_mask_point(x, mask_coarse_logits, fg_boxes, fg_labels,
                                                   fg_roi_masks)
        rcnn_mask_coarse_loss = self.maskrcnn_loss2(mask_coarse_logits, fg_boxes, fg_labels, fg_roi_masks)
        return loss_mask_point, rcnn_mask_coarse_loss

    def roi_mask_point_loss2(self, mask_logits, points_coord, fg_labels, fg_masks):
        '''roi_mask_point_loss2'''
        tmp = [b.shape[0] for b in fg_masks]
        num_boxes = []
        for i in range(1, len(tmp)):
            tmp[i] += tmp[i - 1]
            num_boxes.append(tmp[i-1])
        point_coords_splits = np.split(points_coord, num_boxes)
        gt_mask_logits = []
        for i, gt_mask in enumerate(fg_masks):
            h, w = gt_mask.shape[1:]
            scale = get_tensor([w, h], mindspore.float32)
            points_coord_grid_sample_format = point_coords_splits[i] / scale
            gt_mask = self.op_expanddims(gt_mask, 1)
            gt_mask_logit = point_sample(gt_mask, points_coord_grid_sample_format)
            gt_mask_logits.append(gt_mask_logit)
        concat = P.Concat(axis=0)
        gt_mask_logits = concat(gt_mask_logits)
        fg_labels = concat(fg_labels)
        if gt_mask_logits.shape[0] == 0:
            return mask_logits.sum() * 0
        total_num_masks = mask_logits.shape[0]
        indices = np.arange(total_num_masks)
        mask_logits = mask_logits[indices, fg_labels]
        squeeze = ops.Squeeze(1)
        gt_mask_logits = squeeze(gt_mask_logits)
        bce_loss = nn.BCEWithLogitsLoss()
        point_loss = bce_loss(mask_logits, gt_mask_logits)
        return point_loss

    def maskrcnn_loss2(self, mask_logits, fg_proposals, gt_labels, fg_masks):
        '''maskrcnn_loss2'''
        gt_masks = []
        for i, fg_mask in enumerate(fg_masks):
            mask_target = self.project_masks_on_boxes(fg_mask, fg_proposals[i])
            gt_masks.append(mask_target)
        gt_masks = self.concat(gt_masks)
        gt_labels = self.concat(gt_labels)
        indices = np.arange(gt_labels.shape[0])
        mask_logits = mask_logits[indices, gt_labels]
        gt_masks = self.op_squeeze(gt_masks)
        gt_masks = self.cast(gt_masks, mstype.float32)
        bce_loss = nn.BCEWithLogitsLoss()
        loss_mask_coarse = bce_loss(mask_logits, gt_masks)
        return loss_mask_coarse

@constexpr
def get_tensor(data, datatype):
    return Tensor(data, datatype)
