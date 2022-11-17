# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""centerface networks"""

# from src.config import ConfigCenterface
from src.model_utils.config import config
from src.mobile_v2 import mobilenet_v2
from src.losses import FocalLoss, SmoothL1LossNew, SmoothL1LossNewCMask

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def convTranspose2x2(in_channels, out_channels, has_bias=False): #  Davinci devices only support 'groups=1'
    return nn.Conv2dTranspose(in_channels, out_channels, kernel_size=2, stride=2, has_bias=has_bias,
                              weight_init='normal', bias_init='zeros')


class IDAUp(nn.Cell):
    """
    IDA Module.
    """
    def __init__(self, out_dim, channel):
        super(IDAUp, self).__init__()
        self.out_dim = out_dim
        self.up = nn.SequentialCell([
            convTranspose2x2(out_dim, out_dim, has_bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.9).add_flags_recursive(fp32=True),
            nn.ReLU()])
        self.conv = nn.SequentialCell([
            conv1x1(channel, out_dim),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.9).add_flags_recursive(fp32=True),
            nn.ReLU()])

    def construct(self, x0, x1):
        x = self.up(x0)
        y = self.conv(x1)
        out = x + y
        return out


class MobileNetUp(nn.Cell):
    """
    Mobilenet module.
    """
    def __init__(self, channels, out_dim=24):
        super(MobileNetUp, self).__init__()
        channels = channels[::-1]
        self.conv = nn.SequentialCell([
            conv1x1(channels[0], out_dim),
            nn.BatchNorm2d(out_dim, eps=0.001).add_flags_recursive(fp32=True),
            nn.ReLU()])
        self.conv_last = nn.SequentialCell([
            conv3x3(out_dim, out_dim),
            nn.BatchNorm2d(out_dim, eps=1e-5, momentum=0.99).add_flags_recursive(fp32=True),
            nn.ReLU()])

        self.up1 = IDAUp(out_dim, channels[1])
        self.up2 = IDAUp(out_dim, channels[2])
        self.up3 = IDAUp(out_dim, channels[3])

    def construct(self, x1, x2, x3, x4): # tuple/list can be type of input of a subnet
        x = self.conv(x4)  # top_layer, change outdim

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.conv_last(x)
        return x

class Cast(nn.Cell):
    def __init__(self):
        super(Cast, self).__init__()
        self.cast = P.Cast()

    def construct(self, x):
        return self.cast(x, ms.float32)

class CenterfaceMobilev2(nn.Cell):
    """
    Mobilev2 based CenterFace network.

    Args:
        num_classes: Integer. Class number.
        feature_shape: List. Input image shape, [N,C,H,W].

    Returns:
        Cell, cell instance of Darknet based YOLOV3 neural network.
        CenterFace use the same structure.

    Examples:
        yolov3_darknet53(80, [1,3,416,416])

    """

    def __init__(self):
        super(CenterfaceMobilev2, self).__init__()
        self.config = config

        self.base = mobilenet_v2()
        channels = self.base.feat_channel
        self.dla_up = MobileNetUp(channels, out_dim=self.config.head_conv)

        self.hm_head = nn.SequentialCell([conv1x1(self.config.head_conv, 1, has_bias=True),
                                          nn.Sigmoid().add_flags_recursive(fp32=True)])
        self.wh_head = conv1x1(self.config.head_conv, 2, has_bias=True)
        self.off_head = conv1x1(self.config.head_conv, 2, has_bias=True)
        self.kps_head = conv1x1(self.config.head_conv, 10, has_bias=True)

    def construct(self, x):
        x1, x2, x3, x4 = self.base(x)
        x = self.dla_up(x1, x2, x3, x4)

        output_hm = self.hm_head(x)
        output_wh = self.wh_head(x)
        output_off = self.off_head(x)
        output_kps = self.kps_head(x)
        return output_hm, output_wh, output_off, output_kps

class CenterFaceLoss(nn.Cell):
    """
    Loss method definition.
    """
    def __init__(self, wh_weight, reg_offset, off_weight, hm_weight, lm_weight):
        super(CenterFaceLoss, self).__init__()
        # --- config parameter
        self.wh_weight = wh_weight
        self.reg_offset = reg_offset
        self.off_weight = off_weight
        self.hm_weight = hm_weight
        self.lm_weight = lm_weight
        # ---
        self.cls_loss = FocalLoss()
        self.reg_loss = SmoothL1LossNew()
        self.reg_loss_cmask = SmoothL1LossNewCMask()

    def construct(self, output_hm, output_wh, output_off, output_kps, hm, reg_mask, ind, wh, wight_mask, hm_offset,
                  hps_mask, landmarks):
        """
        Construct method.
        """
        hm_loss = self.cls_loss(output_hm, hm)  # 1. focal loss, center points
        wh_loss = self.reg_loss(output_wh, ind, wh, wight_mask)  # 2. weight and height
        off_loss = self.reg_loss(output_off, ind, hm_offset, wight_mask)  # 3. offset
        lm_loss = self.reg_loss_cmask(output_kps, hps_mask, ind, landmarks)  # 4. landmark loss

        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
               self.off_weight * off_loss + self.lm_weight * lm_loss

        # add print when you want to see loss detail and do debug
        return loss


class CenterFaceWithLossCell(nn.Cell):
    """
    Centerface with loss cell.
    """
    def __init__(self, network):
        super(CenterFaceWithLossCell, self).__init__()
        self.centerface_network = network
        self.config = config
        self.loss = CenterFaceLoss(self.config.wh_weight, self.config.reg_offset, self.config.off_weight,
                                   self.config.hm_weight, self.config.lm_weight)
        self.reduce_sum = P.ReduceSum()

    def construct(self, x, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks):
        output_hm, output_wh, output_off, output_kps = self.centerface_network(x)
        loss = self.loss(output_hm, output_wh, output_off, output_kps, hm, reg_mask, ind, wh, wight_mask, hm_offset,
                         hps_mask, landmarks)
        return loss

class TrainingWrapper(nn.TrainOneStepWithLossScaleCell):
    """
    Training wrapper
    """
    def __init__(self, network, optimizer, sens=1.0):
        scaling_sens = sens
        if isinstance(scaling_sens, (int, float)):
            scaling_sens = ms.Tensor(scaling_sens, ms.float32)
        super(TrainingWrapper, self).__init__(network, optimizer, scaling_sens)
        self.sens = sens
    # x, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks
    def construct(self, x, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks):
        """
        Construct method.
        """
        weights = self.weights
        scaling_sens = self.scale_sense
        loss = self.network(x, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks)
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        #sens = sens_input #P.Fill()(P.DType()(loss), P.Shape()(loss), sens_input) # user can contral loss scale by add a sens_input
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(x, hm, reg_mask, ind, wh, wight_mask, hm_offset, hps_mask, landmarks,
                                                 sens)
        #grads = self.hyper_map(F.partial(_grad_scale, sens), grads) # if add this, the loss_scale optimizer is needed to set to 1
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        ret = (loss, cond, sens)
        self.optimizer(grads)
        return ret


class CenterFaceWithNms(nn.Cell):
    """
    CenterFace with nms.
    """
    def __init__(self, network):
        super(CenterFaceWithNms, self).__init__()
        self.centerface_network = network
        self.config = config
        # two type of maxpool self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.maxpool2d = P.MaxPoolWithArgmax(kernel_size=3, strides=1, pad_mode='same')
        self.topk = P.TopK(sorted=True)
        self.reshape = P.Reshape()
        self.test_batch = self.config.test_batch_size
        self.k = self.config.K

    def construct(self, x):
        """
        Construct method.
        """
        output_hm, output_wh, output_off, output_kps = self.centerface_network(x)
        output_hm_nms, _ = self.maxpool2d(output_hm)
        abs_error = P.Abs()(output_hm - output_hm_nms)
        abs_out = P.Abs()(output_hm)
        error = abs_error / (abs_out + 1e-12)

        # cannot use P.Equal()(output_hm, output_hm_nms), since maxpooling output has 0.1% error
        keep = P.Select()(P.LessEqual()(error, 1e-3), \
           P.Fill()(ms.float32, P.Shape()(error), 1.0), \
           P.Fill()(ms.float32, P.Shape()(error), 0.0))
        output_hm = output_hm * keep

        # get topK and index
        scores = self.reshape(output_hm, (self.test_batch, -1))
        topk_scores, topk_inds = self.topk(scores, self.k)
        return topk_scores, output_wh, output_off, output_kps, topk_inds
