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
"""FasterRcnn training network wrapper."""

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dtype = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dtype),
                                   F.cast(F.tuple_to_array((clip_value,)), dtype))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dtype))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


class StudentNetBurnInWithLoss(nn.Cell):
    def __init__(self, network):
        super(StudentNetBurnInWithLoss, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, label_img_q, label_img_k, label_img_shape, label_gt_bboxes, label_gt_label, label_gt_num):
        label_q_rpn_loss, label_q_rcnn_loss, label_q_rpn_cls_loss, label_q_rpn_reg_loss, \
            label_q_rcnn_cls_loss, label_q_rcnn_reg_loss = self.network(label_img_q, label_img_shape,
                                                                        label_gt_bboxes, label_gt_label, label_gt_num)
        label_k_rpn_loss, label_k_rcnn_loss, label_k_rpn_cls_loss, label_k_rpn_reg_loss, \
            label_k_rcnn_cls_loss, label_k_rcnn_reg_loss = self.network(label_img_k, label_img_shape,
                                                                        label_gt_bboxes, label_gt_label, label_gt_num)

        label_rpn_cls_loss = label_q_rpn_cls_loss + label_k_rpn_cls_loss
        label_rpn_reg_loss = label_q_rpn_reg_loss + label_k_rpn_reg_loss
        label_rcnn_cls_loss = label_q_rcnn_cls_loss + label_k_rcnn_cls_loss
        label_rcnn_reg_loss = label_q_rcnn_reg_loss + label_k_rcnn_reg_loss

        sum_loss = label_q_rpn_loss + label_q_rcnn_loss + label_k_rpn_loss + label_k_rcnn_loss

        label_rpn_cls_loss = F.stop_gradient(label_rpn_cls_loss)
        label_rpn_reg_loss = F.stop_gradient(label_rpn_reg_loss)
        label_rcnn_cls_loss = F.stop_gradient(label_rcnn_cls_loss)
        label_rcnn_reg_loss = F.stop_gradient(label_rcnn_reg_loss)

        return sum_loss, label_rpn_cls_loss, label_rpn_reg_loss, label_rcnn_cls_loss, label_rcnn_reg_loss


class StudentNetBurnInDynamicTrainOneStep(nn.TrainOneStepWithLossScaleCell):

    def construct(self, label_img_q, label_img_k, label_img_shape, label_gt_bboxes, label_gt_label, label_gt_num):
        weights = self.weights
        loss, label_rpn_cls_loss, label_rpn_reg_loss, label_rcnn_cls_loss, label_rcnn_reg_loss = \
            self.network(label_img_q, label_img_k, label_img_shape, label_gt_bboxes, label_gt_label, label_gt_num)

        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        sens_1 = P.Fill()(P.DType()(label_rpn_cls_loss), P.Shape()(label_rpn_cls_loss), 0.0)
        sens_2 = P.Fill()(P.DType()(label_rpn_reg_loss), P.Shape()(label_rpn_reg_loss), 0.0)
        sens_3 = P.Fill()(P.DType()(label_rcnn_cls_loss), P.Shape()(label_rcnn_cls_loss), 0.0)
        sens_4 = P.Fill()(P.DType()(label_rcnn_reg_loss), P.Shape()(label_rcnn_reg_loss), 0.0)

        grads = self.grad(self.network, weights)(label_img_q, label_img_k, label_img_shape, label_gt_bboxes,
                                                 label_gt_label, label_gt_num,
                                                 (scaling_sens, sens_1, sens_2, sens_3, sens_4))

        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            succ = self.optimizer(grads)
        else:
            succ = False
        overflow_flag = F.cast(overflow, mstype.float32)
        ret = (loss, label_rpn_cls_loss, label_rpn_reg_loss, label_rcnn_cls_loss, label_rcnn_reg_loss,
               overflow_flag, scaling_sens)
        return F.depend(ret, succ)


class StudentNetBurnUpWithLoss(nn.Cell):
    def __init__(self, network, unsup_loss_weight):
        super(StudentNetBurnUpWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.unsup_loss_weight = Tensor(np.array(unsup_loss_weight).astype(np.float32))

    def construct(self, label_img_q, label_img_k, label_img_shape, label_gt_bboxes, label_gt_label, label_gt_num,
                  unlabel_img, unlabel_img_shape, unlabel_gt_bboxes, unlabel_gt_label, unlabel_gt_num):
        label_q_rpn_loss, label_q_rcnn_loss, label_q_rpn_cls_loss, label_q_rpn_reg_loss, \
            label_q_rcnn_cls_loss, label_q_rcnn_reg_loss = self.network(label_img_q, label_img_shape,
                                                                        label_gt_bboxes, label_gt_label, label_gt_num)
        label_k_rpn_loss, label_k_rcnn_loss, label_k_rpn_cls_loss, label_k_rpn_reg_loss, \
            label_k_rcnn_cls_loss, label_k_rcnn_reg_loss = self.network(label_img_k, label_img_shape,
                                                                        label_gt_bboxes, label_gt_label, label_gt_num)

        label_rpn_cls_loss = label_q_rpn_cls_loss + label_k_rpn_cls_loss
        label_rpn_reg_loss = label_q_rpn_reg_loss + label_k_rpn_reg_loss
        label_rcnn_cls_loss = label_q_rcnn_cls_loss + label_k_rcnn_cls_loss
        label_rcnn_reg_loss = label_q_rcnn_reg_loss + label_k_rcnn_reg_loss

        unlabel_rpn_loss, unlabel_rcnn_loss, unlabel_rpn_cls_loss, unlabel_rpn_reg_loss, \
            unlabel_rcnn_cls_loss, unlabel_rcnn_reg_loss = self.network(unlabel_img, unlabel_img_shape,
                                                                        unlabel_gt_bboxes, unlabel_gt_label,
                                                                        unlabel_gt_num)

        sum_loss = label_q_rpn_loss + label_q_rcnn_loss + label_k_rpn_loss + label_k_rcnn_loss + \
                   (unlabel_rpn_loss + unlabel_rcnn_loss) * self.unsup_loss_weight

        label_rpn_cls_loss = F.stop_gradient(label_rpn_cls_loss)
        label_rpn_reg_loss = F.stop_gradient(label_rpn_reg_loss)
        label_rcnn_cls_loss = F.stop_gradient(label_rcnn_cls_loss)
        label_rcnn_reg_loss = F.stop_gradient(label_rcnn_reg_loss)
        unlabel_rpn_cls_loss = F.stop_gradient(unlabel_rpn_cls_loss)
        unlabel_rpn_reg_loss = F.stop_gradient(unlabel_rpn_reg_loss)
        unlabel_rcnn_cls_loss = F.stop_gradient(unlabel_rcnn_cls_loss)
        unlabel_rcnn_reg_loss = F.stop_gradient(unlabel_rcnn_reg_loss)

        return sum_loss, label_rpn_cls_loss, label_rpn_reg_loss, label_rcnn_cls_loss, label_rcnn_reg_loss, \
               unlabel_rpn_cls_loss, unlabel_rpn_reg_loss, unlabel_rcnn_cls_loss, unlabel_rcnn_reg_loss


class StudentNetBurnUpDynamicTrainOneStep(nn.TrainOneStepWithLossScaleCell):

    def construct(self, label_img_q, label_img_k, label_img_shape, label_gt_bboxes, label_gt_label, label_gt_num,
                  unlabel_img, unlabel_img_shape, unlabel_gt_bboxes, unlabel_gt_label, unlabel_gt_num):
        weights = self.weights
        loss, label_rpn_cls_loss, label_rpn_reg_loss, label_rcnn_cls_loss, label_rcnn_reg_loss, \
        unlabel_rpn_cls_loss, unlabel_rpn_reg_loss, unlabel_rcnn_cls_loss, unlabel_rcnn_reg_loss = \
            self.network(label_img_q, label_img_k, label_img_shape, label_gt_bboxes, label_gt_label, label_gt_num,
                         unlabel_img, unlabel_img_shape, unlabel_gt_bboxes, unlabel_gt_label, unlabel_gt_num)

        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        sens_1 = P.Fill()(P.DType()(label_rpn_cls_loss), P.Shape()(label_rpn_cls_loss), 0.0)
        sens_2 = P.Fill()(P.DType()(label_rpn_reg_loss), P.Shape()(label_rpn_reg_loss), 0.0)
        sens_3 = P.Fill()(P.DType()(label_rcnn_cls_loss), P.Shape()(label_rcnn_cls_loss), 0.0)
        sens_4 = P.Fill()(P.DType()(label_rcnn_reg_loss), P.Shape()(label_rcnn_reg_loss), 0.0)
        sens_5 = P.Fill()(P.DType()(unlabel_rpn_cls_loss), P.Shape()(unlabel_rpn_cls_loss), 0.0)
        sens_6 = P.Fill()(P.DType()(unlabel_rpn_reg_loss), P.Shape()(unlabel_rpn_reg_loss), 0.0)
        sens_7 = P.Fill()(P.DType()(unlabel_rcnn_cls_loss), P.Shape()(unlabel_rcnn_cls_loss), 0.0)
        sens_8 = P.Fill()(P.DType()(unlabel_rcnn_reg_loss), P.Shape()(unlabel_rcnn_reg_loss), 0.0)

        grads = self.grad(self.network, weights)(label_img_q, label_img_k, label_img_shape, label_gt_bboxes,
                                                 label_gt_label, label_gt_num,
                                                 unlabel_img, unlabel_img_shape, unlabel_gt_bboxes,
                                                 unlabel_gt_label, unlabel_gt_num,
                                                 (scaling_sens, sens_1, sens_2, sens_3, sens_4,
                                                  sens_5, sens_6, sens_7, sens_8))

        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            succ = self.optimizer(grads)
        else:
            succ = False
        overflow_flag = F.cast(overflow, mstype.float32)
        ret = (loss, label_rpn_cls_loss, label_rpn_reg_loss, label_rcnn_cls_loss, label_rcnn_reg_loss,
               unlabel_rpn_cls_loss, unlabel_rpn_reg_loss, unlabel_rcnn_cls_loss, unlabel_rcnn_reg_loss,
               overflow_flag, scaling_sens)
        return F.depend(ret, succ)
