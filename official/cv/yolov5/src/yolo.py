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
"""YOLOv5 based on DarkNet."""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.backbone import YOLOv5Backbone, Conv, BottleneckCSP
from src.loss import ConfidenceLoss, ClassLoss

from model_utils.config import config as default_config

class YOLO(nn.Cell):
    def __init__(self, backbone, shape):
        super(YOLO, self).__init__()
        self.backbone = backbone
        self.config = default_config
        self.config.out_channel = (self.config.num_classes + 5) * 3

        self.conv1 = Conv(shape[5], shape[4], k=1, s=1)
        self.CSP5 = BottleneckCSP(shape[5], shape[4], n=1*shape[6], shortcut=False)
        self.conv2 = Conv(shape[4], shape[3], k=1, s=1)
        self.CSP6 = BottleneckCSP(shape[4], shape[3], n=1*shape[6], shortcut=False)
        self.conv3 = Conv(shape[3], shape[3], k=3, s=2)
        self.CSP7 = BottleneckCSP(shape[4], shape[4], n=1*shape[6], shortcut=False)
        self.conv4 = Conv(shape[4], shape[4], k=3, s=2)
        self.CSP8 = BottleneckCSP(shape[5], shape[5], n=1*shape[6], shortcut=False)
        self.back_block1 = YoloBlock(shape[3], self.config.out_channel)
        self.back_block2 = YoloBlock(shape[4], self.config.out_channel)
        self.back_block3 = YoloBlock(shape[5], self.config.out_channel)

        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """
        input_shape of x is (batch_size, 3, h, w)
        feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        """
        img_height = x.shape[2] * 2
        img_width = x.shape[3] * 2

        feature_map1, feature_map2, feature_map3 = self.backbone(x)

        c1 = self.conv1(feature_map3)
        ups1 = ops.ResizeNearestNeighbor((img_height // 16, img_width // 16))(c1)
        c2 = self.concat((ups1, feature_map2))
        c3 = self.CSP5(c2)
        c4 = self.conv2(c3)
        ups2 = ops.ResizeNearestNeighbor((img_height // 8, img_width // 8))(c4)
        c5 = self.concat((ups2, feature_map1))
        # out
        c6 = self.CSP6(c5)
        c7 = self.conv3(c6)

        c8 = self.concat((c7, c4))
        # out
        c9 = self.CSP7(c8)
        c10 = self.conv4(c9)
        c11 = self.concat((c10, c1))
        # out
        c12 = self.CSP8(c11)
        small_object_output = self.back_block1(c6)
        medium_object_output = self.back_block2(c9)
        big_object_output = self.back_block3(c12)
        return small_object_output, medium_object_output, big_object_output


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv5.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(12, 255)

    """
    def __init__(self, in_channels, out_channels):
        super(YoloBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        """construct method"""

        out = self.conv(x)
        return out

class DetectionBlock(nn.Cell):
    """
     YOLOv5 detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: config, Configuration instance.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32)
     """

    def __init__(self, scale, config=default_config, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
            self.scale_x_y = 1.2
            self.offset_x_y = 0.1
        elif scale == 'm':
            idx = (3, 4, 5)
            self.scale_x_y = 1.1
            self.offset_x_y = 0.05
        elif scale == 'l':
            idx = (6, 7, 8)
            self.scale_x_y = 1.05
            self.offset_x_y = 0.025
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4+1+self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        self.pow = ops.Pow()
        self.transpose = ops.Transpose()
        self.exp = ops.Exp()
        self.conf_training = is_training

    def construct(self, x, input_shape):
        """construct method"""
        num_batch = x.shape[0]
        grid_size = x.shape[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = self.reshape(x, (num_batch,
                                      self.num_anchors_per_scale,
                                      self.num_attrib,
                                      grid_size[0],
                                      grid_size[1]))
        prediction = self.transpose(prediction, (0, 3, 4, 1, 2))

        grid_x = ms.numpy.arange(grid_size[1])
        grid_y = ms.numpy.arange(grid_size[0])
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.scale_x_y * self.sigmoid(box_xy) - self.offset_x_y + grid) / \
                 ops.cast(ops.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = self.exp(box_wh) * self.anchors / input_shape

        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.conf_training:
            return prediction, box_xy, box_wh
        return self.concat((box_xy, box_wh, box_confidence, box_probs))


class Iou(nn.Cell):
    """Calculate the iou of boxes"""
    def __init__(self):
        super(Iou, self).__init__()
        self.min = ops.Minimum()
        self.max = ops.Maximum()
        self.squeeze = ops.Squeeze(-1)

    def construct(self, box1, box2):
        """
        box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        convert to topLeft and rightDown
        """
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / ops.scalar_to_array(2.0) # topLeft
        box1_maxs = box1_xy + box1_wh / ops.scalar_to_array(2.0) # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / ops.scalar_to_array(2.0)
        box2_maxs = box2_xy + box2_wh / ops.scalar_to_array(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, ops.scalar_to_array(0.0))
        # self.squeeze: for effiecient slice
        intersect_area = self.squeeze(intersect_wh[:, :, :, :, :, 0:1]) * \
                         self.squeeze(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = self.squeeze(box1_wh[:, :, :, :, :, 0:1]) * \
                    self.squeeze(box1_wh[:, :, :, :, :, 1:2])
        box2_area = self.squeeze(box2_wh[:, :, :, :, :, 0:1]) * \
                    self.squeeze(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


class YoloLossBlock(nn.Cell):
    """
    Loss block cell of YOLOV5 network.
    """
    def __init__(self, scale, config=default_config):
        super(YoloLossBlock, self).__init__()
        self.config = config
        if scale == 's':
            # anchor mask
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = ms.Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = ops.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.confidence_loss = ConfidenceLoss()
        self.class_loss = ClassLoss()

        self.reduce_sum = ops.ReduceSum()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.reshape = ops.Reshape()
        self.expand_dims = ops.ExpandDims()
        self.ones_like = ops.OnesLike()
        self.log = ops.Log()
        self.tuple_to_array = ops.TupleToArray()
        self.g_iou = GIou()

    def construct(self, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        """
        prediction : origin output from yolo
        pred_xy: (sigmoid(xy)+grid)/grid_size
        pred_wh: (exp(wh)*anchors)/input_shape
        y_true : after normalize
        gt_box: [batch, maxboxes, xyhw] after normalize
        """
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]
        true_boxes = y_true[:, :, :, :, :4]

        grid_shape = prediction.shape[1:3]
        grid_shape = ops.cast(self.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = self.select(self.equal(true_wh, 0.0),
                              self.ones_like(true_wh),
                              true_wh)
        true_wh = self.log(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = gt_box.shape
        gt_box = self.reshape(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(self.expand_dims(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ops.cast(ignore_mask, ms.float32)
        ignore_mask = self.expand_dims(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = ops.stop_gradient(ignore_mask)

        confidence_loss = self.confidence_loss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.class_loss(object_mask, prediction[:, :, :, :, 5:], class_probs)

        object_mask_me = self.reshape(object_mask, (-1, 1))  # [8, 72, 72, 3, 1]
        box_loss_scale_me = self.reshape(box_loss_scale, (-1, 1))
        pred_boxes_me = xywh2x1y1x2y2(pred_boxes)
        pred_boxes_me = self.reshape(pred_boxes_me, (-1, 4))
        true_boxes_me = xywh2x1y1x2y2(true_boxes)
        true_boxes_me = self.reshape(true_boxes_me, (-1, 4))
        c_iou = self.g_iou(pred_boxes_me, true_boxes_me)
        c_iou_loss = object_mask_me * box_loss_scale_me * (1 - c_iou)
        c_iou_loss_me = self.reduce_sum(c_iou_loss, ())
        loss = c_iou_loss_me * 4 + confidence_loss + class_loss
        batch_size = prediction.shape[0]
        return loss / batch_size


class YOLOV5(nn.Cell):
    """
    YOLOV5 network.

    Args:
        is_training: Bool. Whether train or not.

    Returns:
        Cell, cell instance of YOLOV5 neural network.

    Examples:
        YOLOV5s(True)
    """

    def __init__(self, is_training, version=0):
        super(YOLOV5, self).__init__()
        self.config = default_config

        # YOLOv5 network
        self.shape = self.config.input_shape[version]
        self.feature_map = YOLO(backbone=YOLOv5Backbone(shape=self.shape), shape=self.shape)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training)
        self.detect_2 = DetectionBlock('m', is_training=is_training)
        self.detect_3 = DetectionBlock('s', is_training=is_training)

    def construct(self, x, input_shape):
        small_object_output, medium_object_output, big_object_output = self.feature_map(x)
        output_big = self.detect_1(big_object_output, input_shape)
        output_me = self.detect_2(medium_object_output, input_shape)
        output_small = self.detect_3(small_object_output, input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small


class YOLOV5s_Infer(nn.Cell):
    """
    YOLOV5 Infer.
    """

    def __init__(self, input_shape, version=0):
        super(YOLOV5s_Infer, self).__init__()
        self.network = YOLOV5(is_training=False, version=version)
        self.input_shape = input_shape

    def construct(self, x):
        return self.network(x, self.input_shape)


class YoloWithLossCell(nn.Cell):
    """YOLOV5 loss."""
    def __init__(self, network):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = default_config
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)
        self.tenser_to_array = ops.TupleToArray()

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape):
        input_shape = x.shape[2:4]
        input_shape = ops.cast(self.tenser_to_array(input_shape) * 2, ms.float32)

        yolo_out = self.yolo_network(x, input_shape)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        return loss_l + loss_m + loss_s * 0.2


class GIou(nn.Cell):
    """Calculating giou"""
    def __init__(self):
        super(GIou, self).__init__()
        self.reshape = ops.Reshape()
        self.min = ops.Minimum()
        self.max = ops.Maximum()
        self.concat = ops.Concat(axis=1)
        self.mean = ops.ReduceMean()
        self.div = ops.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(ops.cast(intersection, ms.float32), ops.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(ops.cast(res_mid0, ms.float32), ops.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = ops.clip_by_value(giou, -1.0, 1.0)
        return giou


def xywh2x1y1x2y2(box_xywh):
    boxes_x1 = box_xywh[..., 0:1] - box_xywh[..., 2:3] / 2
    boxes_y1 = box_xywh[..., 1:2] - box_xywh[..., 3:4] / 2
    boxes_x2 = box_xywh[..., 0:1] + box_xywh[..., 2:3] / 2
    boxes_y2 = box_xywh[..., 1:2] + box_xywh[..., 3:4] / 2
    boxes_x1y1x2y2 = ops.Concat(-1)((boxes_x1, boxes_y1, boxes_x2, boxes_y2))

    return boxes_x1y1x2y2
