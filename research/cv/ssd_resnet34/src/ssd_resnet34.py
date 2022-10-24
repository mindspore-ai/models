# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""ssd_resnet34"""

from src.resnet34 import resnet34
import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C


class SSD_ResNet34(nn.Cell):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        vggt: pretrained vgg16 (partial) model
        label_num: number of classes (including background 0)
    """

    def __init__(self, config):
        super(SSD_ResNet34, self).__init__()
        self.strides = [1, 1, 2, 2, 2, 1]
        self.module = resnet34()
        out_size = 38
        out_channels = config.extras_out_channels
        self._build_additional_features(out_size, out_channels)
        # init_net_param()

    def _build_additional_features(self, input_size, input_channels):
        """
        Build additional features
        """
        idx = 0
        if input_size == 38:
            idx = 0
        elif input_size == 19:
            idx = 1
        elif input_size == 10:
            idx = 2

        self.additional_blocks = []

        if input_size == 38:
            self.additional_blocks.append(nn.SequentialCell(
                nn.Conv2d(input_channels[idx], 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, input_channels[idx + 1], kernel_size=3, pad_mode='pad', padding=1,
                          stride=self.strides[2]),
                nn.ReLU(),
            ))
            idx += 1

        self.additional_blocks.append(nn.SequentialCell(
            nn.Conv2d(input_channels[idx], 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, input_channels[idx + 1], kernel_size=3, pad_mode='pad', padding=1, stride=self.strides[3]),
            nn.ReLU(),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.SequentialCell(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, input_channels[idx + 1], kernel_size=3, pad_mode='pad', padding=1, stride=self.strides[4]),
            nn.ReLU(),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.SequentialCell(
            nn.Conv2d(input_channels[idx], 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, input_channels[idx + 1], kernel_size=3, pad_mode='valid', stride=self.strides[5]),
            nn.ReLU(),
        ))
        idx += 1

        # Only necessary in VGG for now
        if input_size >= 19:
            # conv11_1, conv11_2
            self.additional_blocks.append(nn.SequentialCell(
                nn.Conv2d(input_channels[idx], 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, input_channels[idx + 1], kernel_size=3, pad_mode='valid'),
                nn.ReLU(),
            ))

        self.additional_blocks = nn.CellList(self.additional_blocks)

    def construct(self, x):
        """
        Construct SSD_ResNet34
        """
        layers = self.module(x)
        # last result from network goes into additional blocks
        layer0 = layers[-1]
        # additional_results = []
        layer1 = self.additional_blocks[0](layer0)
        layers.append(layer1)
        layer2 = self.additional_blocks[1](layer1)
        layers.append(layer2)
        layer3 = self.additional_blocks[2](layer2)
        layers.append(layer3)
        layer4 = self.additional_blocks[3](layer3)
        layers.append(layer4)
        layer5 = self.additional_blocks[4](layer4)
        layers.append(layer5)
        return layers


class SigmoidFocalClassificationLoss(nn.Cell):
    """"
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = F.cast(label, mstype.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSDWithLossCell(nn.Cell):
    """"
    Provide SSD training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): SSD config.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, network, config):
        super(SSDWithLossCell, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        """Construct SSDWithLossCell"""
        pred_loc, pred_label = self.network(x)
        mask = F.cast(self.less(0, gt_label), mstype.float32)
        num_matched_boxes = self.reduce_sum(F.cast(num_matched_boxes, mstype.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_sum(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer, sens=1.0, use_global_norm=False):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = C.HyperMap()

    def construct(self, *args):
        """Construct TrainingWrapper"""
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_tensor(self.sens)), grads)
            grads = C.clip_by_global_norm(grads)
        self.optimizer(grads)
        return loss
