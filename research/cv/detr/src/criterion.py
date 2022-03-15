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
"""criterion"""
import numpy as np

from src import box_ops
from src import grad_ops
from src.matcher import build_matcher


def numpy_one_hot(targets, num_classes):
    """one hot"""
    one_hots = []
    bs, l = targets.shape
    for target in targets:
        one_hot = np.zeros((len(target), num_classes))
        index = np.arange(target.shape[0])
        one_hot[index, target] = 1
        one_hots.append(one_hot)
    return np.concatenate(one_hots).reshape((bs, l, num_classes))


class SetCriterion:
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.empty_weight = np.ones((self.num_classes + 1,), np.float32)
        self.empty_weight[-1] = self.eos_coef

    def softmax_ce_withlogit_withweight(self, logits, target_classes, weight):
        """softmax ce with logits with weight"""
        target_classes_oh = numpy_one_hot(target_classes,
                                          self.num_classes + 1)
        softmax = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
        weighted_target = weight * target_classes_oh
        ce = -(weighted_target * np.log(softmax)).sum(axis=2)
        sum_weight_target = weight[target_classes.astype(np.int32)].sum()
        ce = ce.sum() / sum_weight_target

        ce_grad_src = ((softmax - target_classes_oh) *
                       weighted_target.sum(axis=2, keepdims=True) / sum_weight_target)
        return ce, ce_grad_src

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = np.concatenate([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = np.full(src_logits.shape[:2], self.num_classes, dtype=np.int64)
        target_classes[idx] = target_classes_o

        loss_ce, ce_grad_src = self.softmax_ce_withlogit_withweight(src_logits, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce,
                  'loss_ce_grad_src': ce_grad_src}
        return losses

    @staticmethod
    def loss_cardinality(outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = [len(v["labels"]) for v in targets]

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = np.abs(card_pred - tgt_lengths).mean()
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = np.concatenate([t['boxes'][i] for t, (_, i) in zip(targets, indices)])

        losses = {}

        loss_bbox = np.abs(src_boxes - target_boxes)
        l1_grad_src = grad_ops.grad_l1(src_boxes, target_boxes)
        l1_grad_src_full = np.zeros_like(outputs['pred_boxes'])
        l1_grad_src_full[idx] = l1_grad_src
        losses['loss_bbox'] = np.nan_to_num(loss_bbox / num_boxes, posinf=0.0).sum()
        losses['loss_bbox_grad_src'] = l1_grad_src_full

        giou, giou_grad_src = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
            calc_grad=True
        )
        giou_grad_src_full = np.zeros_like(outputs['pred_boxes'])
        giou_grad_src_full[idx] = giou_grad_src
        loss_giou = 1 - np.diag(giou)
        losses['loss_giou'] = np.nan_to_num(loss_giou.sum() / num_boxes, posinf=0.0)
        losses['loss_giou_grad_src'] = giou_grad_src_full
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = np.concatenate([np.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = np.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = np.concatenate([np.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = np.concatenate([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def __call__(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        new_targets = []
        num_boxes = targets['n_boxes']
        for i, n in enumerate(num_boxes):
            new_targets.append({
                'boxes': targets['bboxes'][i][: n],
                'labels': targets['labels'][i][: n]
            })

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, new_targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in new_targets)
        num_boxes = np.array([num_boxes])

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, new_targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, new_targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, new_targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion(cfg):
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.class_loss_coef,
                   'loss_bbox': cfg.bbox_loss_coef,
                   'loss_giou': cfg.giou_loss_coef}
    if cfg.aux_loss:
        aux_weight_dict = {}
        for i in range(cfg.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        cfg.num_classes,
        matcher,
        weight_dict,
        cfg.eos_coef,
        cfg.losses
    )
    return criterion
