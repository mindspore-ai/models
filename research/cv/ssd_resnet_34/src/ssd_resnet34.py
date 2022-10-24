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

"""ssd-resnet34"""

from typing import List

import easydict
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.composite as C
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.init_params import filter_checkpoint_parameter_by_list
from src.init_params import init_net_param
from src.multi_box import MultiBox
from src.resnet34 import resnet34


class SSDResNet34Layers(nn.Cell):
    """A class for building layers for the SSD-Resnet34 network
    and stacking them into nn.Cell instance.

    Args:
        config (EasyDict): A Dictionary with the model configuration.

    Returns:
        SSD-ResNet34 layers.
    """

    def __init__(self, config: easydict.EasyDict) -> None:
        super(SSDResNet34Layers, self).__init__()
        self.strides = [1, 1, 2, 2, 2, 1]
        self.resnet = resnet34()
        out_size = 38
        out_channels = config.extras_out_channels
        self._build_additional_features(out_size, out_channels)

    def _build_additional_features(
            self,
            input_size: int,
            output_channels: List[int],
    ) -> None:
        """Build SSD blocks for the model construction."""
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
                nn.Conv2d(output_channels[idx], 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(
                    256,
                    output_channels[idx + 1],
                    kernel_size=3,
                    pad_mode='pad',
                    padding=1,
                    stride=self.strides[2],
                ),
                nn.ReLU(),
            ))
            idx += 1

        self.additional_blocks.append(nn.SequentialCell(
            nn.Conv2d(output_channels[idx], 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                256,
                output_channels[idx + 1],
                kernel_size=3,
                pad_mode='pad',
                padding=1,
                stride=self.strides[3],
            ),
            nn.ReLU(),
        ))
        idx += 1

        # conv9_1, conv9_2
        self.additional_blocks.append(nn.SequentialCell(
            nn.Conv2d(output_channels[idx], 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                128,
                output_channels[idx + 1],
                kernel_size=3,
                pad_mode='pad',
                padding=1,
                stride=self.strides[4],
            ),
            nn.ReLU(),
        ))
        idx += 1

        # conv10_1, conv10_2
        self.additional_blocks.append(nn.SequentialCell(
            nn.Conv2d(output_channels[idx], 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                128,
                output_channels[idx + 1],
                kernel_size=3,
                pad_mode='valid',
                stride=self.strides[5],
            ),
            nn.ReLU(),
        ))
        idx += 1

        # Only necessary in VGG for now
        if input_size >= 19:
            # conv11_1, conv11_2
            self.additional_blocks.append(nn.SequentialCell(
                nn.Conv2d(output_channels[idx], 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(
                    128,
                    output_channels[idx + 1],
                    kernel_size=3,
                    pad_mode='valid',
                ),
                nn.ReLU(),
            ))

        self.additional_blocks = nn.CellList(self.additional_blocks)

    def construct(self, x):
        """Build SSD-ResNet34 layers and return them as nn.CellList instance."""
        layers = self.resnet(x)

        # last result from network goes into additional blocks
        layer0 = layers[-1]
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
    """Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples.
        alpha (float): Hyper-parameter to balance the positive and negative example.

    Returns:
        A Focal Loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits: ms.Tensor, label: ms.Tensor):
        """Calculate a Focal Loss

        Args:
            logits: Predicted labels.
            label: A ground truth labels.

        Returns:
            A Focal Loss
        """
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = F.cast(label, mstype.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSDTrainWithLossCell(nn.Cell):
    """A Class for SSD-Resnet34 training loss through network.

    Args:
        config (dict): SSD config.

    Returns:
        (Tensor): the loss of the network.
    """

    def __init__(self, config):
        super(SSDTrainWithLossCell, self).__init__()
        self.ssd_resnet34_network = build_ssd_resnet34_for_train(config)
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        """Get model output and calculate loss for train.

        Args:
            x: Input
            gt_loc: Ground truth locations.
            gt_label: Ground truth labels.
            num_matched_boxes: A number of matched boxes.

        Returns:
            (Tensor): the loss of the network.
        """
        pred_loc, pred_label = self.ssd_resnet34_network(x)
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
    """Downscale gradients with a given scale value.
    The gradient elements will be divided by the scale value.

    Args:
        scale (float): A downscale value.
        grad (Tensor): A Gradient to downscale.

    Returns:
        A downscaled gradient.
    """

    return grad * P.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    """Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (nn.Cell): The training network with added loss function.
        optimizer (nn.Optimizer): Optimizer for updating the weights.
        sens (float): The adjust parameter.
        use_grad_clip (bool): Whether apply global norm before optimizer.
        pretrain_path (str): Path to the checkpoint to load.

    Returns:
        Loss value.
    """

    def __init__(
            self,
            network: nn.Cell,
            optimizer: nn.Optimizer,
            sens: float = 1.0,
            use_grad_clip: bool = False,
            pretrain_path: str = None,
    ):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.pretrain_path = pretrain_path

        if self.pretrain_path:
            param_dict = load_checkpoint(self.pretrain_path)
            # Delete old learning rate from dict for correct loading in new opt
            filter_checkpoint_parameter_by_list(param_dict, ['learning_rate'])

            load_param_into_net(network, param_dict, True)
            load_param_into_net(optimizer, param_dict, True)

        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_grad_clip = use_grad_clip
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
        """Calculate loss and update weights.

        Args:
            *args: The arguments of the network.

        Returns:
            Loss value.
        """
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.use_grad_clip:
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_tensor(self.sens)), grads)
            grads = C.clip_by_global_norm(grads)
        self.optimizer(grads)
        return loss


class SSDResNet34Model(nn.Cell):
    """A class with SSD-ResNet34 network to predict locations and labels.

    Args:
        config (EasyDict): A Dictionary with model configuration.

    Returns:
        pred_loc (Tensor): The prediction locations.
        pred_label (Tensor): The prediction labels.
    """

    def __init__(self, config: easydict.EasyDict):
        super(SSDResNet34Model, self).__init__()
        self.multi_box = MultiBox(config)
        self.activation = P.Sigmoid()
        self.feature_extractor = SSDResNet34Layers(config)

    def construct(self, x):
        """Extract features from input and return location and label.

        Args:
            x: Input.

        Returns:
            pred_loc (Tensor): The prediction locations.
            pred_label (Tensor): The prediction labels.
        """
        features = self.feature_extractor(x)
        pred_loc, pred_label = self.multi_box(features)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, mstype.float32)
        pred_label = F.cast(pred_label, mstype.float32)
        return pred_loc, pred_label


class SSDInferWithDecoder(nn.Cell):
    """SSD-Resnet34 Inference wrapper to decode the bbox locations.

    Args:
        default_boxes (Tensor): The default_boxes from anchor generator.
        config (EasyDict): A Dictionary with the model configuration.

    Returns:
        pred_xy (Tensor): The locations for bbox after decoder representing (y0, x0, y1, x1).
        pred_label (Tensor): The prediction labels.
    """

    def __init__(
            self,
            default_boxes: ms.Tensor,
            config: easydict.EasyDict,
    ):
        super(SSDInferWithDecoder, self).__init__()
        self.network = SSDResNet34Model(config)
        self.default_boxes = default_boxes
        self.prior_scaling_xy = config.prior_scaling[0]
        self.prior_scaling_wh = config.prior_scaling[1]

    def construct(self, x):
        """Get network predictions and decode the bbox locations.

        Args:
            x: Input

        Returns:
            pred_xy (Tensor): The locations for bbox after decoder representing (y0, x0, y1, x1).
            pred_label (Tensor): The prediction labels.
        """
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = P.Exp()(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = P.Concat(-1)((pred_xy_0, pred_xy_1))
        pred_xy = P.Maximum()(pred_xy, 0)
        pred_xy = P.Minimum()(pred_xy, 1)

        return pred_xy, pred_label


def _load_feature_extractor(ssd: SSDResNet34Model, config: easydict.EasyDict) -> None:
    """Load Resnet34 backbone checkpoint and rename old batch-norm names.

    Args:
        ssd (SSDResNet34Model): nn.Cell instance of SSD-ResNet34 network.
        config (EasyDict): A Dictionary with the model configuration.

    Returns:
        None
    """
    replaced_bn_count = 0
    if config.feature_extractor_base_param:
        param_dict = load_checkpoint(config.feature_extractor_base_param)
        for x in list(param_dict.keys()):
            # Replace old bn 1.2 naming
            key = x
            if "bn1d" in x:
                key = key.replace("bn1d", "bn1")
                replaced_bn_count += 1
            elif "bn2d" in x:
                key = key.replace("bn2d", "bn2")
                replaced_bn_count += 1
            param_dict["feature_extractor.resnet." + key] = param_dict[x]
            del param_dict[x]
        load_param_into_net(ssd.feature_extractor.resnet, param_dict)
        print("Replaced {} batchnorm keys for correct loading resnet backbone.".format(replaced_bn_count))


def build_ssd_resnet34_for_train(config: easydict.EasyDict) -> SSDResNet34Model:
    """Build the SSD-Resnet34 model for training, load pretrained ResNet backbone
    and init parameters of the network.

    Args:
        config (EasyDict): A Dictionary with the model configuration.

    Returns:
        ssd_resnet_model (SSDResNet34Model): SSD-Resnet34 network with prepared for training parameters.
    """
    if config.model == "ssd-resnet34":
        ssd_resnet_model = SSDResNet34Model(config=config)

        # If we resume training from checkpoint, we do not need to load resnet34 pretrained backbone.
        if not config.pre_trained:
            print("Loading ResNet34 pretrained backbone from {}".format(config.feature_extractor_base_param))
            init_net_param(ssd_resnet_model)
            _load_feature_extractor(ssd_resnet_model, config)
    else:
        raise ValueError("config.model: {} is not supported".format(config.model))

    return ssd_resnet_model
