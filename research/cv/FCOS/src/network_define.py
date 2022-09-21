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
"""network_define"""
import sys
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode

from src.config import DefaultConfig

def coords_fmap2orig(feature, stride):
    """
    transform one feature map coords to orig coords
    Args
    feature [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    """
    h, w = feature.shape[1:3]
    shifts_x = mnp.arange(start=0, stop=w * stride, step=stride)
    shifts_y = mnp.arange(start=0, stop=h * stride, step=stride)
    shift_x, shift_y = mnp.meshgrid(shifts_x, shifts_y)
    shift_x = mnp.reshape(shift_x, -1)
    shift_y = mnp.reshape(shift_y, -1)
    coords = mnp.stack((shift_x, shift_y), -1) + stride // 2
    return ops.Cast()(coords, mstype.float32)


class GenTargets(nn.Cell):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
    def getTargets(self, inputs):
        """
        inputs
        [0]tuple (cls_logits,cnt_logits,reg_preds)
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        [1]gt_boxes [batch_size,m,4]  FloatTensor
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        """

        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = ()
        cnt_targets_all_level = ()
        reg_targets_all_level = ()
        for level in range(len(cls_logits)):
            level_out = (cls_logits[level], cnt_logits[level], reg_preds[level])
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level = cls_targets_all_level + (level_targets[0],)
            cnt_targets_all_level = cnt_targets_all_level + (level_targets[1],)
            reg_targets_all_level = reg_targets_all_level + (level_targets[2],)

        return ops.Concat(axis=1)(cls_targets_all_level), ops.Concat(axis=1)(cnt_targets_all_level), ops.Concat(axis=1)(
            reg_targets_all_level)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
        gt_boxes [batch_size,m,4]
        classes [batch_size,m]
        stride int
        limit_range list [min,max]
        Returns
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]
        transpose = ops.Transpose()
        reshape = ops.Reshape()
        cls_logits = transpose(cls_logits, (0, 2, 3, 1))  # [batch_size,h,w,class_num]
        coords = coords_fmap2orig(cls_logits, stride)  # [h*w,2]
        cls_logits = reshape(cls_logits, (batch_size, -1, class_num))  # [batch_size,h*w,class_num]
        cnt_logits = transpose(cnt_logits, (0, 2, 3, 1))
        cnt_logits = reshape(cnt_logits, (batch_size, -1, 1))
        reg_preds = transpose(reg_preds, (0, 2, 3, 1))
        reg_preds = reshape(reg_preds, (batch_size, -1, 4))
        x = coords[:, 0]
        y = coords[:, 1]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = ops.Stack(axis=-1)((l_off, t_off, r_off, b_off))  # [batch_size,h*w,m,4]
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]
        off_min = mnp.amin(ltrb_off, axis=-1)
        off_max = mnp.amax(ltrb_off, axis=-1)
        mask_in_gtboxes = off_min > 0
        tempmin = off_max > limit_range[0]
        tempmax = off_max <= limit_range[1]
        tempmin = ops.Cast()(tempmin, mindspore.int32)
        tempmax = ops.Cast()(tempmax, mindspore.int32)
        tempMask_in_level = ops.Mul()(tempmin, tempmax)
        mask_in_level = ops.Cast()(tempMask_in_level, mindspore.bool_)

        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = ops.Stack(axis=-1)((c_l_off, c_t_off, c_r_off, c_b_off))  # [batch_size,h*w,m,4]
        c_off_max = mnp.amax(c_ltrb_off, axis=-1)
        mask_center = c_off_max < radiu
        tempingtboxes = ops.Cast()(mask_in_gtboxes, mindspore.int32)
        tempmaskinlevel = ops.Cast()(mask_in_level, mindspore.int32)
        tempmaskcenter = ops.Cast()(mask_center, mindspore.int32)
        mask_pos = ops.Mul()(ops.Mul()(tempingtboxes, tempmaskinlevel), tempmaskcenter)
        mask_pos = ops.Cast()(mask_pos, mstype.bool_)
        areas[~mask_pos] = 99999999
        tempareas = areas.reshape(-1, areas.shape[-1])
        areas_min_ind = P.ArgMinWithValue(-1)(tempareas)
        x = mnp.arange(0, areas_min_ind[0].shape[0]).astype(mindspore.int32)
        indices = P.Concat(-1)((P.ExpandDims()(x, -1), P.ExpandDims()(areas_min_ind[0], -1)))
        reg_targets = P.GatherNd()(ltrb_off.reshape(-1, m, 4), indices)
        reg_targets = ops.Reshape()(reg_targets, (batch_size, -1, 4))
        classes = mnp.broadcast_to(classes[:, None, :], areas.shape)
        cls_targets = P.GatherNd()(classes.reshape(-1, m), indices)
        cls_targets = ops.Reshape()(cls_targets, (batch_size, -1, 1))
        # [batch_size,h*w]
        left_right_min = ops.Minimum()(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = ops.Maximum()(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = ops.Minimum()(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = ops.Maximum()(reg_targets[..., 1], reg_targets[..., 3])
        # [batch_size,h*w,1]
        cnt_targets = ops.Sqrt()((left_right_min * top_bottom_min + 1e-8) / (left_right_max * top_bottom_max + 1e-8))
        cnt_targets = ops.ExpandDims()(cnt_targets, -1)
        mask_pos_2 = ops.Cast()(mask_pos, mstype.float16)
        mask_pos_2 = ops.ReduceSum()(mask_pos_2, -1)
        mask_pos_2 = mask_pos_2 >= 1
        expand_dims = ops.ExpandDims()
        mask_pos_2 = expand_dims(mask_pos_2, 2)
        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]
        cnt_targets[~mask_pos_2] = -1
        stack = ops.Stack(axis=2)
        tempmask = ()
        i = 4
        while i:
            i -= 1
            tempmask += (mask_pos_2,)
        mask_pos_2 = stack(tempmask)
        squeeze = ops.Squeeze(3)
        mask_pos_2 = squeeze(mask_pos_2)
        reg_targets[~mask_pos_2] = -1
        return cls_targets, cnt_targets, reg_targets

def compute_cls_loss(preds, targets, mask, MIN, MAX):
    '''
    Args
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    preds_reshape = ()
    class_num = preds[0].shape[1]
    mask = ops.ExpandDims()(mask, -1)
    # [batch_size,]
    mask = ops.Cast()(mask, mstype.float32)
    num_pos = ops.ReduceSum()(mask, (1, 2))
    ones = ops.Ones()
    candidate = ones(num_pos.shape, mindspore.float32)
    num_pos = mnp.where(num_pos == 0, candidate, num_pos)
    num_pos = ops.Cast()(num_pos, mstype.float32)
    for pred in preds:
        pred = ops.Transpose()(pred, (0, 2, 3, 1))
        pred = ops.Reshape()(pred, (batch_size, -1, class_num))
        preds_reshape = preds_reshape + (pred,)
    preds = ops.Concat(axis=1)(preds_reshape)
    loss = ()
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]
        ar = mnp.arange(1, class_num + 1).astype(mstype.float32)
        ar = ar[None, :]
        target_pos = (ar == target_pos)
        # sparse-->onehot
        target_pos = ops.Cast()(target_pos, mstype.float32)
        fl_result = focal_loss_from_logits(pred_pos, target_pos)
        fl_result = ops.Reshape()(fl_result, (1,))
        loss = loss + (fl_result,)
    # [batch_size,]
    return ops.Concat()(loss) / num_pos

def compute_cnt_loss(preds, targets, mask, MIN, MAX):
    '''
    Args
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]:Tensor(Bool)
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = ()
    mask = ops.ExpandDims()(mask, -1)
    mask = ops.Cast()(mask, mstype.float32)
    num_pos = ops.ReduceSum()(mask, (1, 2))
    ones = ops.Ones()
    candidate = ones(num_pos.shape, mindspore.float32)
    num_pos = mnp.where(num_pos == 0, candidate, num_pos)
    num_pos = ops.Cast()(num_pos, mstype.float32)
    for pred in preds:
        pred = P.Transpose()(pred, (0, 2, 3, 1))
        pred = P.Reshape()(pred, (batch_size, -1, c))
        preds_reshape = preds_reshape + (pred,)
    preds = P.Concat(axis=1)(preds_reshape)
    loss = ()
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index].flatten()
        target_pos = targets[batch_index].flatten()
        weight = P.Ones()(pred_pos.shape, mstype.float32)
        pred_pos = P.Sigmoid()(pred_pos)
        if pred_pos.shape[0] != 0:
            bce_result = nn.BCELoss(weight=weight, reduction='none')(pred_pos, target_pos)
            a = bce_result
            b = ops.Squeeze(1)(mask[batch_index])
            c = ops.Mul()(a, b)
            op = ops.ReduceSum()
            bce_result = op(c)
        else:
            bce_result = mnp.zeros((1,), mindspore.float32)
        bce_result = P.Reshape()(bce_result, (1,))
        loss += (bce_result,)
    return P.Concat(axis=0)(loss) / num_pos

def compute_reg_loss(preds, targets, mask, MIN, MAX, ZERO, ZB, mode='giou'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = ()
    mask = ops.Cast()(mask, mstype.float32)
    num_pos = ops.ReduceSum()(mask, (1,))
    ones = ops.Ones()
    candidate = ones(num_pos.shape, mindspore.float32)
    num_pos = mnp.where(num_pos == 0, candidate, num_pos)
    num_pos = ops.Cast()(num_pos, mstype.float32)
    for pred in preds:
        pred = ops.Transpose()(pred, (0, 2, 3, 1))
        pred = ops.Reshape()(pred, (batch_size, -1, c))
        preds_reshape = preds_reshape + (pred,)
    preds = ops.Concat(axis=1)(preds_reshape)
    loss = ()
    for batch_index in range(batch_size):
        mask_index = mask[batch_index]
        pred_pos = preds[batch_index]
        target_pos = targets[batch_index]
        if pred_pos.shape[0] != 0:
            loss_result = giou_loss(pred_pos, target_pos, mask_index, ZERO, ZB, MAX)
        else:
            loss_result = mnp.zeros((1,), mindspore.float32)
        loss_result = loss_result.reshape((1,))
        loss = loss + (loss_result,)
    return ops.Concat()(loss) / num_pos

def giou_loss(preds, targets, mask_index, ZERO, ZB, MAX):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    minimum = ops.Minimum()
    maximum = ops.Maximum()
    lt_min = minimum(preds[:, :2], targets[:, :2])
    rb_min = minimum(preds[:, 2:], targets[:, 2:])
    wh_min = rb_min + lt_min
    zeros = ops.Zeros()
    candidate = zeros(wh_min.shape, mindspore.float32)
    wh_min = mnp.where(wh_min < 0, candidate, wh_min)
    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union
    lt_max = maximum(preds[:, :2], targets[:, :2])
    rb_max = maximum(preds[:, 2:], targets[:, 2:])
    wh_max = rb_max + lt_max
    zeros = ops.Zeros()
    candidate = zeros(wh_max.shape, mindspore.float32)
    wh_max = mnp.where(wh_max < 0, candidate, wh_max)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]
    zeros = ops.Zeros()
    candidate = zeros(G_area.shape, mindspore.float32) + 1e-10
    G_area = mnp.where(G_area <= 0, candidate, G_area)
    giou = iou - (G_area - union) / G_area  # back3
    loss = (1. - giou).reshape(1, -1)
    mask_index = mask_index.reshape(-1, 1)
    loss = ops.Cast()(loss, mstype.float32)
    loss = ops.dot(loss, mask_index)
    return loss

def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    '''
    Args:
    preds: [n,class_num]
    targets: [n,class_num]
    '''
    preds = ops.Sigmoid()(preds)
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * ops.Pow()((1.0 - pt), gamma) * ops.Log()(pt)
    return ops.ReduceSum()(loss)

class LossNet(nn.Cell):
    """loss method"""
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config
        self.min_value = Tensor(1.)
        self.max_value = Tensor(sys.maxsize, mstype.float32)
        self.zero = Tensor(0.)
        self.zerobottom = Tensor(1e-10)

    def construct(self, inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        mask_pos = ops.Squeeze(axis=-1)(cnt_targets > -1)  # [batch_size,sum(_h*_w)]
        mean = ops.ReduceMean()
        cls_loss = mean(compute_cls_loss(cls_logits, cls_targets, mask_pos, self.min_value, self.max_value))
        cnt_loss = mean(compute_cnt_loss(cnt_logits, cnt_targets, mask_pos, self.min_value, self.max_value))
        reg_loss = mean(compute_reg_loss(reg_preds, reg_targets, mask_pos, self.min_value, self.max_value, \
        self.zero, self.zerobottom))
        cls_loss = ops.Reshape()(cls_loss, (1,))
        cnt_loss = ops.Reshape()(cnt_loss, (1,))
        reg_loss = ops.Reshape()(reg_loss, (1,))
        total_loss = cls_loss + cnt_loss + reg_loss
        return total_loss


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        config = DefaultConfig
        self._backbone = backbone
        self._targets_fn = GenTargets(config.strides, config.limit_range)
        self._loss_fn = loss_fn

    def construct(self, input_imgs, input_boxes, input_classes):
        #preds
        out = self._backbone(input_imgs)
        # stop gradients
        cls_logits = ()
        cnt_logits = ()
        reg_preds = ()
        temp_cls_logits, temp_cnt_logits, temp_reg_preds = out
        for i in temp_cls_logits:
            cls_logits = cls_logits + (ops.Zeros()(i.shape, i.dtype),)
        for i in temp_cnt_logits:
            cnt_logits = cnt_logits + (ops.Zeros()(i.shape, i.dtype),)
        for i in temp_reg_preds:
            reg_preds = reg_preds + (ops.Zeros()(i.shape, i.dtype),)
        stop_out = (cls_logits, cnt_logits, reg_preds)
        targets = self._targets_fn.getTargets((stop_out, input_boxes, input_classes))
        return self._loss_fn((out, targets))

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone

class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        reduce_flag (bool): The reduce flag. Default value is False.
        mean (bool): Allreduce method. Default value is False.
        degree (int): Device number. Default value is None.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, input_imgs, input_boxes, input_classes):
        loss = self.network(input_imgs, input_boxes, input_classes)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(input_imgs, input_boxes, input_classes, sens)
        grads = C.clip_by_global_norm(grads, clip_norm=3.0)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
