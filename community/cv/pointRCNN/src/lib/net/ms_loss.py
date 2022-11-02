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
"""mindspore"""
import mindspore as ms
from mindspore import nn, ops
from src.lib.config import cfg
import src.lib.utils.loss_utils as loss_utils


class net_with_loss(nn.Cell):
    """Net Loss"""
    def __init__(self, backbone, cols_name):
        super().__init__()
        self._backbone = backbone
        self.cols_name = cols_name

    def construct(self, *cols):
        """construct function"""
        data = dict(zip(self.cols_name, cols))
        if cfg.RPN.ENABLED:
            _, _, pts_input = data['pts_rect'], data['pts_features'], data[
                'pts_input']
            gt_boxes3d = data['gt_boxes3d']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data[
                    'rpn_reg_label']

                rpn_cls_label = rpn_cls_label.astype(ms.int32)
                rpn_reg_label = rpn_reg_label.astype(ms.float32)

            inputs = pts_input.astype(ms.float32)
            gt_boxes3d = gt_boxes3d.astype(ms.float32)
            input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d}
        else:
            input_data = {}
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = ms.Tensor.from_numpy(
                        val).contiguous().astype(ms.float32)
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = ops.concat(
                    (input_data['pts_input'], input_data['pts_features']),
                    axis=-1)
                input_data['pts_input'] = pts_input

        ret_dict = self._backbone(input_data)

        tb_dict = {}
        disp_dict = {}
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
            rpn_loss = self.get_rpn_loss(rpn_cls, rpn_reg, rpn_cls_label,
                                         rpn_reg_label, tb_dict)
            loss += rpn_loss.asnumpy().item()
            disp_dict['rpn_loss'] = rpn_loss.asnumpy().item()

        if cfg.RCNN.ENABLED:
            rcnn_loss = self.get_rcnn_loss(ret_dict, tb_dict)
            disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']
            loss += rcnn_loss.asnumpy().item()

        disp_dict['loss'] = loss

        return loss

    def get_rpn_loss(self, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label,
                     tb_dict):
        """get rpn loss"""
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(
                alpha=cfg.RPN.FOCAL_ALPHA[0], gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':

            rpn_cls_loss_func = ops.BinaryCrossEntropy()
        else:
            raise NotImplementedError

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).astype(ms.float32)
            pos = (rpn_cls_label_flat > 0).astype(ms.float32)
            neg = (rpn_cls_label_flat == 0).astype(ms.float32)
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            _min_va = ms.Tensor(1.0, ms.float32)
            cls_weights = cls_weights / ops.clip_by_value(
                pos_normalizer, _min_va)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target,
                                             cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.asnumpy().item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.asnumpy().item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = ms.numpy.ones(rpn_cls_flat.shape[0], ms.float32)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).astype(ms.float32)
            bce = ops.BinaryCrossEntropy('none')
            batch_loss_cls = bce(ops.sigmoid(rpn_cls_flat),
                                 rpn_cls_label_target, weight)
            cls_valid_mask = (rpn_cls_label_flat >= 0).astype(ms.float32)
            _min_va = ms.Tensor(1.0, ms.float32)
            rpn_loss_cls = (batch_loss_cls *
                            cls_valid_mask).sum() / ops.clip_by_value(
                                cls_valid_mask.sum(), _min_va)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.shape[0] * rpn_reg.shape[1]
        fg_sum = fg_mask.astype(ms.int32).sum().asnumpy().item()
        MEAN_SIZE = ms.Tensor.from_numpy(cfg.CLS_MEAN_SIZE[0])
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, _ = \
                loss_utils.get_reg_loss(rpn_reg.view(point_num, -1),
                                        rpn_reg_label.view(point_num, 7),
                                        fg_mask,
                                        loc_scope=cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size=MEAN_SIZE)

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[
            0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({
            'rpn_loss_cls': rpn_loss_cls.asnumpy().item(),
            'rpn_loss_reg': rpn_loss_reg.asnumpy().item(),
            'rpn_loss': rpn_loss.asnumpy().item(),
            'rpn_fg_sum': fg_sum,
            'rpn_loss_loc': loss_loc.asnumpy().item(),
            'rpn_loss_angle': loss_angle.asnumpy().item(),
            'rpn_loss_size': loss_size.asnumpy().item()
        })

        return rpn_loss

    def get_rcnn_loss(self, ret_dict, tb_dict):
        """get rcnn loss"""
        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(
                alpha=cfg.RCNN.FOCAL_ALPHA[0], gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':

            cls_loss_func = ops.BinaryCrossEntropy()
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':

            cls_weight = ms.Tensor.from_numpy(cfg.RCNN.CLS_WEIGHT).astype(
                ms.float32)
            cls_loss_func = nn.CrossEntropyLoss(weight=cls_weight,
                                                ignore_index=-1,
                                                reduce=None)
        else:
            raise NotImplementedError

        MEAN_SIZE = ms.Tensor.from_numpy(cfg.CLS_MEAN_SIZE[0])
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']

        cls_label = ret_dict['cls_label'].astype(ms.float32)
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']

        cls_label_flat = cls_label.view(-1)
        _min_va = ms.Tensor(1.0, ms.float32)
        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).astype(ms.float32)
            pos = (cls_label_flat > 0).astype(ms.float32)
            neg = (cls_label_flat == 0).astype(ms.float32)
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / ops.clip_by_value(
                pos_normalizer, _min_va)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target,
                                          cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.asnumpy().item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.asnumpy().item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = ops.BinaryCrossEntropy('none')(
                ops.sigmoid(rcnn_cls_flat), cls_label,
                ms.numpy.ones_like(cls_label, cls_label.dtype))
            cls_valid_mask = (cls_label_flat >= 0).astype(ms.float32)

            rcnn_loss_cls = (batch_loss_cls *
                             cls_valid_mask).sum() / ops.clip_by_value(
                                 cls_valid_mask.sum(), _min_va)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.astype(ms.int32)
            cls_valid_mask = (cls_label_flat >= 0).astype(ms.float32)

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = ops.clip_by_value(cls_valid_mask.sum(), _min_va)
            rcnn_loss_cls = (batch_loss_cls.mean(axis=1) *
                             cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.astype(ms.int32).sum().asnumpy().item()
        if fg_sum != 0:
            all_anchor_size = roi_size
            anchor_size = all_anchor_size[
                fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        mask=None)

            loss_size = 3 * loss_size  # consistent with old codes
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = rcnn_loss_reg = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.asnumpy().item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.asnumpy().item()
        tb_dict['rcnn_loss'] = rcnn_loss.asnumpy().item()

        tb_dict['rcnn_loss_loc'] = loss_loc.asnumpy().item()
        tb_dict['rcnn_loss_angle'] = loss_angle.asnumpy().item()
        tb_dict['rcnn_loss_size'] = loss_size.asnumpy().item()

        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().asnumpy().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().asnumpy().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().asnumpy().item()

        return rcnn_loss
