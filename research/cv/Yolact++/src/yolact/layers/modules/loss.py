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
"""Calculate the loss function"""
import mindspore
import mindspore.ops as P
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from src.yolact.layers.modules.match import match
from src.yolact.utils.ms_box_utils import center_size, crop
from src.config import yolact_plus_resnet50_config as cfg

class MultiBoxLoss(nn.Cell):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio, batch_size, num_priors):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        self.sum = P.ReduceSum(True)
        self.sum_f = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.concat = P.Concat()
        self.concat2 = P.Concat(2)
        self.squeeze0 = P.Squeeze(0)
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()
        self.max = P.ReduceMax()
        self.transpose = P.Transpose()
        self.softmax = P.Softmax()
        self.exp = P.Exp()
        self.log = P.Log()
        self.topk = P.TopK(sorted=True)
        self.reverse = P.ReverseV2(axis=[1])
        self.oneslike = P.OnesLike()
        self.select = P.Select()
        self.less = P.Less()
        self.matmul = P.MatMul()
        self.cat = P.Concat(2)
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction='sum')
        self.sigmoid = P.Sigmoid()
        self.resize_bilinear = nn.ResizeBilinear()
        self.scalarCast = P.ScalarCast()
        self.smoothL1loss = nn.SmoothL1Loss()
        self.closs = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(reduction='sum')
        self.matmulnn = nn.MatMul(False, True)
        self.gather = P.Gather()
        self.binary_cross_entropy_n = P.BinaryCrossEntropy(reduction='none')
        self.maximum = P.Maximum()
        self.zeroslike = P.ZerosLike()
        self.oneslike = P.OnesLike()
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        self.batch_size = batch_size
        self.num_priors = num_priors

        self.min_value = mindspore.Tensor(0, mindspore.float32)
        self.max_value = mindspore.Tensor(1, mindspore.float32)

        self.match_test = match()
        self.gatherD = P.GatherD()

        self.maskiounet = mask_iou_net()
        self.randChoicewithmask = P.RandomChoiceWithMask(cfg['masks_to_train'])

        self.train_boxes = cfg['train_boxes'] #true
        self.bbox_alpha = cfg['bbox_alpha']
        self.use_segmantic_segmentation_loss = cfg['use_semantic_segmentation_loss'] #true
        self.train_masks = cfg['train_masks'] #true
        self.mask_type = cfg['mask_type'] #{direct 0 limb : 1}
        self.use_maskiou = cfg['use_maskiou'] #true
        self.semantic_segmentation_alpha = cfg['semantic_segmentation_alpha']
        self.conf_alpha = cfg['conf_alpha']
        self.discard_mask_area = cfg['discard_mask_area']
        self.maskiou_alpha = cfg['maskiou_alpha']
        self.mask_proto_binarize_downsampled_gt = cfg['mask_proto_binarize_downsampled_gt'] #true
        self.mask_proto_crop = cfg['mask_proto_crop'] #true
        self.mask_proto_normalize_emulate_roi_pooling = cfg['mask_proto_normalize_emulate_roi_pooling'] #true
        self.mask_alpha = cfg['mask_alpha']


    def construct(self, predictions, gt_bboxes, gt_labels, crowd_boxes, masks):
        """Forward"""
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        priors = predictions['priors']
        proto_data = predictions['proto']
        gt_labels = self.cast(gt_labels, mindspore.int32)

        labels = F.stop_gradient(gt_labels)
        loc_t = ()
        conf_t = ()
        idx_t = ()
        gt_box_t = ()
        for idx in range(self.batch_size):

            truths = gt_bboxes[idx]
            labels_idx = labels[idx]
            truths = F.stop_gradient(truths)
            priors_t = F.stop_gradient(priors)
            priors_t = self.cast(priors_t, mindspore.float32)
            cur_crowd_boxes = crowd_boxes[idx]
            loc_idx, conf_idx, idx_idx = self.match_test(
                self.pos_threshold, self.neg_threshold, truths, priors_t, labels_idx, cur_crowd_boxes)
            loc_t += (self.expand_dims(loc_idx, 0),)
            conf_t += (self.expand_dims(conf_idx, 0),)
            idx_t += (self.expand_dims(idx_idx, 0),)
            gt_box_t += (self.expand_dims(truths[idx_idx], 0),)

        loc_t = self.concat(loc_t)
        conf_t = self.concat(conf_t)
        idx_t = self.concat(idx_t)
        gt_box_t = self.concat(gt_box_t)
        pos = conf_t > 0
        pos = self.cast(pos, mindspore.float32)
        num_pos = self.sum_f(pos)
        num_pos = F.stop_gradient(num_pos)
        pos = self.cast(pos, mindspore.bool_)
        pos_idx = self.expand_dims(self.cast(pos, mindspore.int32), 2).expand_as(loc_data)
        pos_idx = self.cast(pos_idx, mindspore.bool_)
        loss_B = 0
        pos_idx = self.cast(pos_idx, mindspore.float32)
        loss_B = self.sum_f((self.smoothL1loss(loc_data, loc_t) * pos_idx)) * self.bbox_alpha / num_pos

        loss_S = 0
        loss_S = self.semantic_segmentation_loss(predictions['segm'], masks, labels)

        # Mask loss
        maskiou_net_input = 0
        maskiou_t = 0
        label_t = 0
        mask_t = 0
        ret = 0

        ret = self.lincomb_mask_loss(pos, idx_t, mask_data, proto_data, masks, gt_box_t, labels)

        loss_M, maskiou_net_input, maskiou_t, label_t, mask_t = ret

        loss_I = 0
        loss_I = self.mask_iou_loss(maskiou_net_input, maskiou_t, label_t, mask_t, num_pos)

        loss_C = self.ohem_conf_loss(conf_data, conf_t, pos, self.batch_size)


        return loss_B, loss_C, loss_S, loss_M, loss_I


    def semantic_segmentation_loss(self, segment_data, mask_t, class_t):
        """segmentation loss"""
        batch_size, _, mask_h, mask_w = segment_data.shape
        loss_s = 0
        for idx in range(batch_size):
            cur_segment = self.squeeze0(segment_data[idx:idx+1:1, :, :, :])
            cur_class_t = class_t[idx]
            em = mask_t[idx:idx+1:1, :, :, :]
            cm = self.cast(em, mindspore.float32)
            downsampled_masks = self.resize_bilinear(cm, (self.scalarCast(mask_h, mindspore.int32),
                                                          self.scalarCast(mask_w, mindspore.int32)))
            downsampled_masks = self.squeeze0(downsampled_masks)
            downsampled_masks = self.cast(self.less(0.5, downsampled_masks), mindspore.float32)
            segment_t = self.zeroslike(cur_segment)

            # structure segment_t
            for obj_idx in range(downsampled_masks.shape[0]):
                i = self.cast(cur_class_t[obj_idx], mindspore.int32)
                j = self.select(i > 0, i, i + 2)
                segment_t[j, :, :] = self.maximum(segment_t[j, :, :], downsampled_masks[obj_idx, :, :])
                segment_t = F.stop_gradient(segment_t)
            loss_s += self.bcewithlogitsloss(cur_segment, segment_t)

        return loss_s / mask_h / mask_w * self.semantic_segmentation_alpha / batch_size

    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        """Classification loss"""
        batch_conf = conf_data.view(-1, self.num_classes)

        x_max = self.max(self.max(batch_conf, 1), 0)
        loss_c = self.log(self.sum(self.exp(batch_conf - x_max), 1)) + x_max - batch_conf[::, 0:1:1]
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
        loss_c[conf_t < 0] = 0
        k = 57744
        loss_idx = self.topk(loss_c, k)
        _, idx_rank = self.topk(self.cast(loss_idx[1], mindspore.float32), k)
        idx_rank = self.reverse(idx_rank)
        m = self.cast(pos, mindspore.float32)
        num_pos = self.sum(m, 1)
        num_neg = self.negpos_ratio * num_pos

        neg_shape = idx_rank.shape
        broadcast_to1 = P.BroadcastTo(neg_shape)
        num_neg = self.cast(num_neg, mindspore.int32)
        new = broadcast_to1(num_neg)
        neg = idx_rank < new

        pos = self.cast(pos, mindspore.bool_)
        neg = self.cast(neg, mindspore.float32)
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos] = 0
        neg[conf_t < 0] = 0

        pos = self.cast(pos, mindspore.float32)
        c_conf_t = self.less(0, (pos + neg)).view(-1)
        c_conf_t = self.cast(c_conf_t, mindspore.float32)
        all_c_conf_t = self.sum_f(c_conf_t)
        cp = conf_data.view(-1, self.num_classes)
        ct = conf_t.view(-1)
        loss_final = self.sum_f(self.closs(cp, ct) * c_conf_t) / all_c_conf_t

        return self.conf_alpha * loss_final

    def _mask_iou(self, mask1, mask2):

        intersection = self.sum_f(mask1 * mask2, (0, 1))
        area1 = self.sum_f(mask1, (0, 1))
        area2 = self.sum_f(mask2, (0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, maskiou_net_input, maskiou_t, label_t, mask_t, num_pos):
        """Iou loss"""
        maskiou_net_input = self.cast(maskiou_net_input, mindspore.float32)
        perm = (2, 0, 1)
        mask_t = self.transpose(mask_t, perm)
        gt_mask_area = self.sum_f(mask_t, (1, 2))
        select = gt_mask_area > self.discard_mask_area
        select = self.cast(select, mindspore.float32)
        maskiou_p = self.maskiounet(maskiou_net_input)
        label_t = self.expand_dims(label_t, 1)
        gather = self.gatherD(maskiou_p, 1, label_t)
        maskiou_p = gather.view(-1)
        loss_i = self.sum_f(self.smoothL1loss(maskiou_p, maskiou_t) * select)
        loss_i /= num_pos


        return loss_i * self.maskiou_alpha


    def lincomb_mask_loss(self, pos, idx_t, mask_data, proto_data, masks, gt_box_t, labels):
        """Mask loss"""
        mask_h = proto_data.shape[1]
        mask_w = proto_data.shape[2]
        loss_m = 0
        maskiou_t_list = ()
        maskiou_net_input_list = ()
        label_t_list = ()
        mask_t_list = ()
        sums = 0
        for idx in range(mask_data.shape[0]):

            down_mask = self.cast(self.expand_dims(masks[idx], 0), mindspore.float32)
            downsampled_masks = self.resize_bilinear(down_mask, size=(
                self.scalarCast(mask_h, mindspore.int32), self.scalarCast(mask_w, mindspore.int32)),
                                                     align_corners=False)
            downsampled_masks = self.squeeze0(downsampled_masks)
            downsampled_masks = F.stop_gradient(downsampled_masks)

            perm = (1, 2, 0)
            downsampled_masks = self.transpose(downsampled_masks, perm)
            downsampled_masks = self.cast(self.less(0.5, downsampled_masks), mindspore.float32)

            pos = self.cast(pos, mindspore.bool_)
            cur_pos = pos[idx]
            out_idx, _ = self.randChoicewithmask(cur_pos)
            out_idx = out_idx.view(-1)
            out_idx = out_idx.view(-1)
            cur_pos = self.cast(cur_pos, mindspore.float32)
            pos_mask = cur_pos[out_idx]
            num = self.sum_f(pos_mask)
            pos_idx_t = idx_t[idx, out_idx]
            proto_masks = self.squeeze0(proto_data[idx:idx+1:1, ::, ::, ::])
            proto_coef = self.squeeze0(mask_data[idx:idx+1:1, out_idx, ::])
            pos_gt_box_t = self.squeeze0(gt_box_t[idx:idx+1:1, out_idx, ::])
            mask_ = downsampled_masks[:, :, pos_idx_t]

            mask_t_list += (mask_,)
            label_t_ = labels[idx]
            label_t = label_t_[pos_idx_t]

            proto_coef = self.squeeze(proto_coef)
            pred_masks = self.matmulnn(proto_masks, proto_coef)
            pred_masks = self.sigmoid(pred_masks)
            pred_masks = crop(pred_masks, pos_gt_box_t)

            weight_bce = None
            pred_masks = C.clip_by_value(pred_masks, self.min_value, self.max_value)
            pred_masks = self.cast(pred_masks, mindspore.float32)
            pre_loss = self.binary_cross_entropy_n(pred_masks, mask_, weight_bce)
            weight = mask_h * mask_w if self.mask_proto_crop  else 1
            pos_gt_csize = center_size(pos_gt_box_t)
            gt_box_width = pos_gt_csize[:, 2] * mask_w
            gt_box_height = pos_gt_csize[:, 3] * mask_h
            pre_loss = self.sum_f(pre_loss, (0, 1))
            pre_loss = pre_loss / gt_box_width / gt_box_height * weight


            pre_loss = pre_loss * pos_mask
            sums += num
            loss_m += self.sum_f(pre_loss)
            loss_m /= sums

            perm = (2, 0, 1)
            maskiou_net_input = self.expand_dims(self.transpose(pred_masks, perm), 1)
            pred_masks = self.cast(self.less(0.5, pred_masks), mindspore.float32)
            maskiou_t = self._mask_iou(pred_masks, mask_)
            maskiou_net_input_list += (maskiou_net_input,)
            maskiou_t_list += (maskiou_t,)
            label_t_list += (label_t,)

        lossm = loss_m * self.mask_alpha / mask_h / mask_w


        maskiou_t = self.concat(maskiou_t_list)
        label_t = self.concat(label_t_list)
        maskiou_net_input = self.concat(maskiou_net_input_list)
        mask_t = self.concat2(mask_t_list)
        return lossm, maskiou_net_input, maskiou_t, label_t, mask_t


class mask_iou_net(nn.Cell):
    """Build the network"""
    def __init__(self):
        super(mask_iou_net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(2, 2), pad_mode='same',
                               has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), pad_mode='same',
                               has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), pad_mode='same',
                               has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), pad_mode='same',
                               has_bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), pad_mode='same',
                               has_bias=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=80, kernel_size=(1, 1), stride=(1, 1), pad_mode='same',
                               has_bias=True)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d((3, 3), (3, 3), 'valid')
        self.squeeze = P.Squeeze(-1)

    def construct(self, x):
        """Forward"""
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)
        maskiou_p = self.squeeze(self.squeeze(self.max_pool2d(out)))

        return maskiou_p
