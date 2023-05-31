# Copyright 2023 Huawei Technologies Co., Ltd
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
# =======================================================================================
""" yolox module """
import os
import importlib
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import ops, numpy, Tensor, Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .boxes import batch_bboxes_iou
from .network_blocks import BaseConv, DWConv
from .yolo_fpn import YOLOFPN
from .yolo_pafpn import YOLOPAFPN
from .adaptive_module import AdaptiveModule


class DetectionPerFPN(nn.Cell):
    """ head  """

    def __init__(self, num_classes, scale, in_channels=None, act="silu", width=1.0, depthwise=False):
        super(DetectionPerFPN, self).__init__()
        if in_channels is None:
            in_channels = [1024, 512, 256]
        self.scale = scale
        self.num_classes = num_classes
        Conv = DWConv if depthwise else BaseConv

        if scale == 's':
            self.stem = BaseConv(in_channels=int(in_channels[0] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        elif scale == 'm':
            self.stem = BaseConv(in_channels=int(in_channels[1] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        elif scale == 'l':
            self.stem = BaseConv(in_channels=int(in_channels[2] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")

        self.cls_convs = nn.SequentialCell(
            [
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
            ]
        )
        self.reg_convs = nn.SequentialCell(
            [
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
            ]
        )
        self.cls_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=self.num_classes, kernel_size=1, stride=1,
                                   pad_mode="pad", has_bias=True)

        self.reg_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1,
                                   pad_mode="pad",
                                   has_bias=True)

        self.obj_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1,
                                   pad_mode="pad",
                                   has_bias=True)

    def construct(self, x):
        """ forward """
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_convs(cls_x)

        cls_output = self.cls_preds(cls_feat)

        reg_feat = self.reg_convs(reg_x)
        reg_output = self.reg_preds(reg_feat)
        obj_output = self.obj_preds(reg_feat)

        return cls_output, reg_output, obj_output


class DetectionBlock(nn.Cell):
    """ connect yolox backbone and head """

    def __init__(self, config=None, backbone="yolopafpn"):
        super(DetectionBlock, self).__init__()

        if config is None:
            current_path = os.path.realpath(os.path.dirname(__file__))
            config = importlib.import_module(os.path.join(current_path, 'config.py'))
        self.num_classes = config.num_classes
        self.attr_num = self.num_classes + 5
        self.depthwise = config.depth_wise
        self.strides = Tensor([16, 32, 64], mindspore.float32)
        self.input_size = config.input_size

        self.tmm = AdaptiveModule(in_ch=3, nf=config.nf, gamma_range=config.gamma_range)

        # network
        if backbone == "yolopafpn":
            self.depth = 0.33
            self.width = 0.25
            self.backbone = YOLOPAFPN(depth=self.depth,
                                      width=self.width,
                                      input_w=self.input_size[1],
                                      input_h=self.input_size[0],
                                      depthwise=self.depthwise)
            self.head_inchannels = [1024, 512, 256]
            self.activation = "silu"
        else:
            self.backbone = YOLOFPN(input_w=self.input_size[1], input_h=self.input_size[0])
            self.head_inchannels = [512, 256, 128]
            self.activation = "lrelu"
            self.width = 1.0

        self.head_l = DetectionPerFPN(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='l',
                                      act=self.activation, width=self.width, depthwise=self.depthwise)
        self.head_m = DetectionPerFPN(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='m',
                                      act=self.activation, width=self.width, depthwise=self.depthwise)
        self.head_s = DetectionPerFPN(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='s',
                                      act=self.activation, width=self.width, depthwise=self.depthwise)

    def construct(self, x):
        """ forward """
        x_tm = self.tmm(x)
        x_tm = numpy.clip(x_tm, 0, 1) * 255.

        outputs = []
        x_l, x_m, x_s = self.backbone(x_tm)
        cls_output_l, reg_output_l, obj_output_l = self.head_l(x_l)  # (bs, 80, 80, 80)(bs, 4, 80, 80)(bs, 1, 80, 80)
        cls_output_m, reg_output_m, obj_output_m = self.head_m(x_m)  # (bs, 80, 40, 40)(bs, 4, 40, 40)(bs, 1, 40, 40)
        cls_output_s, reg_output_s, obj_output_s = self.head_s(x_s)  # (bs, 80, 20, 20)(bs, 4, 20, 20)(bs, 1, 20, 20)
        if self.training:
            output_l = P.Concat(axis=1)((reg_output_l, obj_output_l, cls_output_l))  # (bs, 85, 80, 80)
            output_m = P.Concat(axis=1)((reg_output_m, obj_output_m, cls_output_m))  # (bs, 85, 40, 40)
            output_s = P.Concat(axis=1)((reg_output_s, obj_output_s, cls_output_s))  # (bs, 85, 20, 20)

            output_l = self.mapping_to_img(output_l, stride=self.strides[0])  # (bs, 6400, 85)x_c, y_c, w, h
            output_m = self.mapping_to_img(output_m, stride=self.strides[1])  # (bs, 1600, 85)x_c, y_c, w, h
            output_s = self.mapping_to_img(output_s, stride=self.strides[2])  # (bs,  400, 85)x_c, y_c, w, h

        else:

            output_l = P.Concat(axis=1)(
                (reg_output_l, P.Sigmoid()(obj_output_l), P.Sigmoid()(cls_output_l)))  # bs, 85, 80, 80

            output_m = P.Concat(axis=1)(
                (reg_output_m, P.Sigmoid()(obj_output_m), P.Sigmoid()(cls_output_m)))  # bs, 85, 40, 40

            output_s = P.Concat(axis=1)(
                (reg_output_s, P.Sigmoid()(obj_output_s), P.Sigmoid()(cls_output_s)))  # bs, 85, 20, 20
            output_l = self.mapping_to_img(output_l, stride=self.strides[0])  # (bs, 6400, 85)x_c, y_c, w, h
            output_m = self.mapping_to_img(output_m, stride=self.strides[1])  # (bs, 1600, 85)x_c, y_c, w, h
            output_s = self.mapping_to_img(output_s, stride=self.strides[2])  # (bs,  400, 85)x_c, y_c, w, h
        outputs.append(output_l)
        outputs.append(output_m)
        outputs.append(output_s)
        return P.Concat(axis=1)(outputs)  # batch_size, 8400, 85

    def mapping_to_img(self, output, stride):
        """ map to origin image scale for each fpn """
        batch_size = P.Shape()(output)[0]
        n_ch = self.attr_num
        grid_size = P.Shape()(output)[2:4]
        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        stride = P.Cast()(stride, output.dtype)
        grid_x = P.Cast()(F.tuple_to_array(range_x), output.dtype)
        grid_y = P.Cast()(F.tuple_to_array(range_y), output.dtype)
        grid_y = P.ExpandDims()(grid_y, 1)
        grid_x = P.ExpandDims()(grid_x, 0)
        yv = P.Tile()(grid_y, (1, grid_size[1]))
        xv = P.Tile()(grid_x, (grid_size[0], 1))
        grid = P.Stack(axis=2)([xv, yv])  # (80, 80, 2)
        grid = P.Reshape()(grid, (1, 1, grid_size[0], grid_size[1], 2))  # (1,1,80,80,2)
        output = P.Reshape()(output,
                             (batch_size, n_ch, grid_size[0], grid_size[1]))  # bs, 6400, 85-->(bs,85,80,80)
        output = P.Transpose()(output, (0, 2, 1, 3))  # (bs,85,80,80)-->(bs,80,85,80)
        output = P.Transpose()(output, (0, 1, 3, 2))  # (bs,80,85,80)--->(bs, 80, 80, 85)
        output = P.Reshape()(output, (batch_size, 1 * grid_size[0] * grid_size[1], -1))  # bs, 6400, 85
        grid = P.Reshape()(grid, (1, -1, 2))  # grid(1, 6400, 2)

        # reconstruct
        output_xy = output[..., :2]
        output_xy = (output_xy + grid) * stride
        output_wh = output[..., 2:4]
        output_wh = P.Exp()(output_wh) * stride
        output_other = output[..., 4:]
        output_t = P.Concat(axis=-1)([output_xy, output_wh, output_other])
        return output_t  # bs, 6400, 85           grid(1, 6400, 2)


class YOLOLossCell(nn.Cell):
    """ yolox with loss cell """

    def __init__(self, network=None, config=None):
        super(YOLOLossCell, self).__init__()
        self.network = network
        self.n_candidate_k = config.n_candidate_k
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.depth = config.num_classes

        self.unsqueeze = P.ExpandDims()
        self.reshape = P.Reshape()
        self.one_hot = P.OneHot()
        self.zeros = P.ZerosLike()
        self.sort_ascending = P.Sort(descending=False)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")
        self.batch_iter = Tensor(np.arange(0, config.per_batch_size * config.max_gt), mindspore.int32)
        self.strides = config.fpn_strides
        self.grids = [(config.input_size[0] // _stride) * (config.input_size[1] // _stride) for _stride in
                      config.fpn_strides]
        self.use_l1 = config.use_l1
        self.use_summary = config.use_summary
        self.summary = ops.ScalarSummary()
        self.assign = ops.Assign()

    def construct(self, img, labels=None, pre_fg_mask=None, is_inbox_and_incenter=None):
        """ forward with loss return """
        batch_size = P.Shape()(img)[0]
        gt_max = P.Shape()(labels)[1]
        outputs = self.network(img)  # batch_size, 8400, 85
        total_num_anchors = P.Shape()(outputs)[1]
        bbox_preds = outputs[:, :, :4]  # batch_size, 8400, 4

        obj_preds = outputs[:, :, 4:5]  # batch_size, 8400, 1
        cls_preds = outputs[:, :, 5:]  # (batch_size, 8400, 80)

        # process label
        bbox_true = labels[:, :, 1:]  # (batch_size, gt_max, 4)

        gt_classes = F.cast(labels[:, :, 0:1].squeeze(-1), mindspore.int32)
        pair_wise_ious = batch_bboxes_iou(bbox_true, bbox_preds, xyxy=False)
        pair_wise_ious = pair_wise_ious * pre_fg_mask
        pair_wise_iou_loss = -P.Log()(pair_wise_ious + 1e-8) * pre_fg_mask
        gt_classes_ = self.one_hot(gt_classes, self.depth, self.on_value, self.off_value)
        gt_classes_expaned = ops.repeat_elements(self.unsqueeze(gt_classes_, 2), rep=total_num_anchors, axis=2)
        gt_classes_expaned = F.stop_gradient(gt_classes_expaned)
        cls_preds_ = P.Sigmoid()(ops.repeat_elements(self.unsqueeze(cls_preds, 1), rep=gt_max, axis=1)) * \
                     P.Sigmoid()(
                         ops.repeat_elements(self.unsqueeze(obj_preds, 1), rep=gt_max, axis=1)
                     )

        pair_wise_cls_loss = P.ReduceSum()(
            P.BinaryCrossEntropy(reduction="none")(P.Sqrt()(cls_preds_), gt_classes_expaned, None), -1)
        pair_wise_cls_loss = pair_wise_cls_loss * pre_fg_mask
        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        punishment_cost = 1000.0 * (1.0 - F.cast(is_inbox_and_incenter, mindspore.float32))
        cost = F.cast(cost + punishment_cost, mindspore.float16)
        # dynamic k matching
        ious_in_boxes_matrix = pair_wise_ious  # (batch_size, gt_max, 8400)
        ious_in_boxes_matrix = F.cast(pre_fg_mask * ious_in_boxes_matrix, mindspore.float16)
        topk_ious, _ = P.TopK(sorted=True)(ious_in_boxes_matrix, self.n_candidate_k)

        dynamic_ks = P.ReduceSum()(topk_ious, 2).astype(mindspore.int32).clip(xmin=1, xmax=total_num_anchors - 1,
                                                                              dtype=mindspore.int32)

        dynamic_ks_indices = P.Stack(axis=1)((self.batch_iter, dynamic_ks.reshape((-1,))))

        dynamic_ks_indices = F.stop_gradient(dynamic_ks_indices)

        values, _ = P.TopK(sorted=True)(-cost, self.n_candidate_k)  # b_s , 50, 8400
        values = P.Reshape()(-values, (-1, self.n_candidate_k))
        max_neg_score = self.unsqueeze(P.GatherNd()(values, dynamic_ks_indices).reshape(batch_size, -1), 2)
        pos_mask = F.cast(cost < max_neg_score, mindspore.float32)  # (batch_size, gt_num, 8400)
        pos_mask = pre_fg_mask * pos_mask
        # ----dynamic_k---- END-----------------------------------------------------------------------------------------
        cost_t = cost * pos_mask + (1.0 - pos_mask) * 2000.
        min_index, _ = P.ArgMinWithValue(axis=1)(cost_t)
        ret_posk = P.Transpose()(nn.OneHot(depth=gt_max, axis=-1)(min_index), (0, 2, 1))
        pos_mask = pos_mask * ret_posk
        pos_mask = F.stop_gradient(pos_mask)
        # AA problem--------------END ----------------------------------------------------------------------------------

        # calculate target ---------------------------------------------------------------------------------------------
        # Cast precision
        pos_mask = F.cast(pos_mask, mindspore.float16)
        bbox_true = F.cast(bbox_true, mindspore.float16)
        gt_classes_ = F.cast(gt_classes_, mindspore.float16)

        reg_target = P.BatchMatMul(transpose_a=True)(pos_mask, bbox_true)  # (batch_size, 8400, 4)
        pred_ious_this_matching = self.unsqueeze(P.ReduceSum()((ious_in_boxes_matrix * pos_mask), 1), -1)
        cls_target = P.BatchMatMul(transpose_a=True)(pos_mask, gt_classes_)

        cls_target = cls_target * pred_ious_this_matching
        obj_target = P.ReduceMax()(pos_mask, 1)  # (batch_size, 8400)

        # calculate l1_target
        reg_target = F.stop_gradient(reg_target)
        cls_target = F.stop_gradient(cls_target)
        obj_target = F.stop_gradient(obj_target)
        bbox_preds = F.cast(bbox_preds, mindspore.float32)
        reg_target = F.cast(reg_target, mindspore.float32)
        obj_preds = F.cast(obj_preds, mindspore.float32)
        obj_target = F.cast(obj_target, mindspore.float32)
        cls_preds = F.cast(cls_preds, mindspore.float32)
        cls_target = F.cast(cls_target, mindspore.float32)
        loss_l1 = 0.0
        if self.use_l1:
            l1_target = self.get_l1_format(reg_target)
            l1_preds = self.get_l1_format(bbox_preds)
            l1_target = F.stop_gradient(l1_target)
            l1_target = F.cast(l1_target, mindspore.float32)
            l1_preds = F.cast(l1_preds, mindspore.float32)
            loss_l1 = P.ReduceSum()(self.l1_loss(l1_preds, l1_target), -1) * obj_target
            loss_l1 = P.ReduceSum()(loss_l1)
        # calculate target -----------END-------------------------------------------------------------------------------
        loss_iou = IOUloss()(P.Reshape()(bbox_preds, (-1, 4)), reg_target).reshape(batch_size, -1) * obj_target
        loss_iou = P.ReduceSum()(loss_iou)
        loss_obj = self.bce_loss(P.Reshape()(obj_preds, (-1, 1)), P.Reshape()(obj_target, (-1, 1)))
        loss_obj = P.ReduceSum()(loss_obj)

        loss_cls = P.ReduceSum()(self.bce_loss(cls_preds, cls_target), -1) * obj_target
        loss_cls = P.ReduceSum()(loss_cls)

        num_fg_mask = P.ReduceSum()(obj_target) == 0
        num_fg = (num_fg_mask == 0) * P.ReduceSum()(obj_target) + 1.0 * num_fg_mask
        loss_all = (5 * loss_iou + loss_cls + loss_obj + loss_l1) / num_fg

        if self.use_summary:
            self.summary('num_fg', num_fg)
            self.summary('loss_iou', loss_iou * 5 / num_fg)
            self.summary('loss_cls', loss_cls / num_fg)
            self.summary('loss_obj', loss_obj / num_fg)
            self.summary('loss_l1', loss_l1 / num_fg)

        return loss_all

    def get_l1_format_single(self, reg_target, stride, eps):
        """ calculate L1 loss related """
        reg_target = reg_target / stride
        reg_target_xy = reg_target[:, :, :2]
        reg_target_wh = reg_target[:, :, 2:]
        reg_target_wh = P.Log()(reg_target_wh + eps)
        return P.Concat(-1)((reg_target_xy, reg_target_wh))

    def get_l1_format(self, reg_target, eps=1e-8):
        """ calculate L1 loss related """
        reg_target_l = reg_target[:, 0:self.grids[0], :]  # (bs, 6400, 4)
        reg_target_m = reg_target[:, self.grids[0]:self.grids[1] + self.grids[0], :]  # (bs, 1600, 4)
        reg_target_s = reg_target[:, -self.grids[2]:, :]  # (bs, 400, 4)

        reg_target_l = self.get_l1_format_single(reg_target_l, self.strides[0], eps)
        reg_target_m = self.get_l1_format_single(reg_target_m, self.strides[1], eps)
        reg_target_s = self.get_l1_format_single(reg_target_s, self.strides[2], eps)

        l1_target = P.Concat(axis=1)([reg_target_l, reg_target_m, reg_target_s])
        return l1_target


class IOUloss(nn.Cell):
    """ Iou loss """

    def __init__(self, reduction="none"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.reshape = P.Reshape()

    def construct(self, pred, target):
        """ forward """
        pred = self.reshape(pred, (-1, 4))
        target = self.reshape(target, (-1, 4))
        tl = P.Maximum()(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        br = P.Minimum()(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        area_p = (pred[:, 2:3] * pred[:, 3:4]).squeeze(-1)
        area_g = (target[:, 2:3] * target[:, 3:4]).squeeze(-1)
        en = F.cast((tl < br), tl.dtype)
        en = (en[:, 0:1] * en[:, 1:2]).squeeze(-1)
        area_i = br - tl
        area_i = (area_i[:, 0:1] * area_i[:, 1:2]).squeeze(-1) * en
        area_u = area_p + area_g - area_i

        iou = area_i / (area_u + 1e-16)
        loss = 1 - iou * iou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class TrainOneStepWithEMA(nn.TrainOneStepWithLossScaleCell):
    """ Train one step with ema model """

    def __init__(self, network, optimizer, sens=1.0, ema=True, decay=0.9998, updates=0):
        super(TrainOneStepWithEMA, self).__init__(network, optimizer, sens)
        self.ema = ema
        self.decay = decay
        self.updates = Parameter(Tensor(updates, mindspore.float32))
        if self.ema:
            self.ema_weight = self.weights.clone("ema", init='same')
            self.moving_parameter = list()
            self.ema_moving_parameter = list()
            self.assign = ops.Assign()
            self.get_moving_parameters()

    def get_moving_parameters(self):
        for key, param in self.network.parameters_and_names():
            if "moving_mean" in key or "moving_variance" in key:
                new_param = param.clone()
                new_param.name = "ema." + param.name
                self.moving_parameter.append(param)
                self.ema_moving_parameter.append(new_param)
        self.moving_parameter = ParameterTuple(self.moving_parameter)
        self.ema_moving_parameter = ParameterTuple(self.ema_moving_parameter)

    def ema_update(self):
        """Update EMA parameters."""
        if self.ema:
            self.updates += 1
            d = self.decay * (1 - ops.Exp()(-self.updates / 2000))
            # update trainable parameters
            for ema_v, weight in zip(self.ema_weight, self.weights):
                tep_v = ema_v * d
                self.assign(ema_v, (1.0 - d) * weight + tep_v)

            for ema_moving, moving in zip(self.ema_moving_parameter, self.moving_parameter):
                tep_m = ema_moving * d
                self.assign(ema_moving, (1.0 - d) * moving + tep_m)
        return self.updates

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        if self.ema:
            self.ema_update()

        # if there is no overflow, do optimize
        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens
