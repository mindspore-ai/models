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
"""Model script."""
import math

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from src.layers import conv_bn
from src.layers import pred
from src.modules import ASPP
from src.modules import DepthwiseM2OIndexBlock
from src.modules import IndexedUpsamlping
from src.modules import InvertedResidual


class MobileNetV2UNetDecoderIndexLearning(nn.Cell):
    """
    IndexNet with MobileNetV2 backbone, UNet architecture.

    Args:
        encoder_rate: Encoder rate.
        encoder_current_stride: Encoder stride.
        encoder_settings: Encoder blocks settings.
        output_stride (int): Output image stride.
        width_mult (float): Width multiplication for mobilenetv2 blocks.
        conv_operator (str): Conv operator for decoder.
        decoder_kernel_size (int): Decoder conv kernel size.
        apply_aspp (bool): Use ASPP.
        use_nonlinear (bool): Use nonlinear in index blocks.
        use_context (bool): Use context in index blocks.
    """
    def __init__(
            self,
            encoder_rate,
            encoder_current_stride,
            encoder_settings,
            output_stride=32,
            width_mult=1.,
            conv_operator='std_conv',
            decoder_kernel_size=5,
            apply_aspp=True,
            use_nonlinear=True,
            use_context=True,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.output_stride = output_stride
        self.encoder_settings = encoder_settings

        # ENCODER
        # building the first layer
        initial_channel = int(self.encoder_settings[0][1] * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 1)
        current_stride = encoder_current_stride * 2
        # building bottleneck layers
        for i, setting in enumerate(self.encoder_settings):
            s = setting[4]
            self.encoder_settings[i][4] = 1  # change stride
            if current_stride == output_stride:
                rate = encoder_rate * s
                self.encoder_settings[i][5] = rate
            else:
                current_stride *= s

        self.layer1 = self._build_layer(InvertedResidual, self.encoder_settings[0])
        self.layer2 = self._build_layer(InvertedResidual, self.encoder_settings[1], downsample=True)
        self.layer3 = self._build_layer(InvertedResidual, self.encoder_settings[2], downsample=True)
        self.layer4 = self._build_layer(InvertedResidual, self.encoder_settings[3], downsample=True)
        self.layer5 = self._build_layer(InvertedResidual, self.encoder_settings[4])
        self.layer6 = self._build_layer(InvertedResidual, self.encoder_settings[5], downsample=True)
        self.layer7 = self._build_layer(InvertedResidual, self.encoder_settings[6])

        # freeze backbone batch norm layers
        self.freeze_bn()

        # define index blocks
        self.index0 = DepthwiseM2OIndexBlock(32, use_nonlinear, use_context)
        self.index2 = DepthwiseM2OIndexBlock(24, use_nonlinear, use_context)
        self.index3 = DepthwiseM2OIndexBlock(32, use_nonlinear, use_context)
        self.index4 = DepthwiseM2OIndexBlock(64, use_nonlinear, use_context)
        self.index6 = DepthwiseM2OIndexBlock(160, use_nonlinear, use_context)

        # context aggregation
        if apply_aspp:
            self.dconv_pp = ASPP(320, 160, output_stride=output_stride, width_mult=width_mult)
        else:
            self.dconv_pp = conv_bn(320, 160, k=1, s=1)

        # DECODER
        self.decoder_layer6 = IndexedUpsamlping(160 * 2, 96, conv_operator, decoder_kernel_size)
        self.decoder_layer5 = IndexedUpsamlping(96 * 2, 64, conv_operator, decoder_kernel_size)
        self.decoder_layer4 = IndexedUpsamlping(64 * 2, 32, conv_operator, decoder_kernel_size)
        self.decoder_layer3 = IndexedUpsamlping(32 * 2, 24, conv_operator, decoder_kernel_size)
        self.decoder_layer2 = IndexedUpsamlping(24 * 2, 16, conv_operator, decoder_kernel_size)
        self.decoder_layer1 = IndexedUpsamlping(16 * 2, 32, conv_operator, decoder_kernel_size)
        self.decoder_layer0 = IndexedUpsamlping(32 * 2, 32, conv_operator, decoder_kernel_size)

        self.pred = pred(32, 1, conv_operator, decoder_kernel_size)

        self.avg_pool = ops.AvgPool(pad_mode='same', kernel_size=(2, 2), strides=(2, 2))

        self._initialize_weights()

    def _build_layer(self, block, layer_setting, downsample=False):
        """
        Build MobileNetV2 block.

        Args:
            block: Encoder block.
            layer_setting (list): Encoder block settings.
            downsample (bool): Downsample at this block.

        Returns:
            block: Inited encoder block.
        """
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t))
            input_channel = output_channel

        return nn.SequentialCell([*layers])

    def _initialize_weights(self):
        """Init model weights."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                weight = np.random.normal(loc=0, scale=math.sqrt(2. / n), size=cell.weight.shape)
                cell.weight.set_data(Tensor(weight, mstype.float32))

    def freeze_bn(self):
        """Freeze batch norms."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.beta.requires_grad = False
                cell.gamma.requires_grad = False
                cell.use_batch_statistics = False

    def construct(self, x):
        """
        Feed forward.

        Args:
            x: Input 4 channel image.

        Returns:
            alpha_channel: Predicted alpha mask.
        """
        # encode
        l0 = self.layer0(x)  # 4x320x320 (for default crop size=320)
        idx0_en, idx0_de = self.index0(l0)
        l0 = idx0_en * l0
        l0p = 4 * self.avg_pool(l0)  # 32x160x160

        l1 = self.layer1(l0p)  # 16x160x160
        l2 = self.layer2(l1)  # 24x160x160
        idx2_en, idx2_de = self.index2(l2)
        l2 = idx2_en * l2
        l2p = 4 * self.avg_pool(l2)  # 24x80x80

        l3 = self.layer3(l2p)  # 32x80x80
        idx3_en, idx3_de = self.index3(l3)
        l3 = idx3_en * l3
        l3p = 4 * self.avg_pool(l3)  # 32x40x40

        l4 = self.layer4(l3p)  # 64x40x40
        idx4_en, idx4_de = self.index4(l4)
        l4 = idx4_en * l4
        l4p = 4 * self.avg_pool(l4)  # 64x20x20

        l5 = self.layer5(l4p)  # 96x20x20
        l6 = self.layer6(l5)  # 160x20x20
        idx6_en, idx6_de = self.index6(l6)
        l6 = idx6_en * l6
        l6p = 4 * self.avg_pool(l6)  # 160x10x10

        l7 = self.layer7(l6p)  # 320x10x10

        # pyramid pooling
        l_up = self.dconv_pp(l7)  # 160x10x10

        # decode
        l_up = self.decoder_layer6(l_up, l6, idx6_de)
        l_up = self.decoder_layer5(l_up, l5)
        l_up = self.decoder_layer4(l_up, l4, idx4_de)
        l_up = self.decoder_layer3(l_up, l3, idx3_de)
        l_up = self.decoder_layer2(l_up, l2, idx2_de)
        l_up = self.decoder_layer1(l_up, l1)
        l_up = self.decoder_layer0(l_up, l0, idx0_de)

        alpha_channel = self.pred(l_up)

        return alpha_channel


class LossWrapper(nn.Cell):
    """
    Train wrapper to the model.

    Args:
        model (nn.Cell): Prediction model.
        loss_function (func): Loss computation between ground-truth and predictions.
    """
    def __init__(self, model, loss_function):
        super().__init__()
        self.model = model
        self.weighted_loss = loss_function

    def construct(
            self,
            inp,
            mask_gt,
            alpha_gt,
            foreground_gt,
            background_gt,
            merged_gt,
    ):
        """
        Get predictions and compute loss.

        Args:
            inp: Input 4 channel image.
            mask_gt: Image mask.
            alpha_gt: Image trimap.
            foreground_gt: Original foreground image.
            background_gt: Original background image.
            merged_gt: Merged by mask foreground over background.

        Returns:
            loss: Computed weighted loss.
        """
        pd = self.model(inp)
        loss = self.weighted_loss(pd, mask_gt, alpha_gt, foreground_gt, background_gt, merged_gt)

        return loss
