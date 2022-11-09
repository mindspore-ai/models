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
import math

import mindspore.nn as nn
from mindspore.common import initializer as init

import src.freia.framework as Ff
import src.freia.modules as Fm

def subnet_conv_func(kernel_size, hidden_ratio, padding):
    """Subnet Convolutional Function.

    Callable class or function ``f``, called as ``f(channels_in, channels_out)`` and
        should return a torch.nn.Module.
        Predicts coupling coefficients :math:`s, t`.

    Args:
        kernel_size (int): Kernel Size
        hidden_ratio (float): Hidden ratio to compute number of hidden channels.

    Returns:
        Callable: Sequential for the subnet constructor.
    """
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.SequentialCell(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding, pad_mode='pad', has_bias=True),
        )

    return subnet_conv

def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    """Create NF Fast Flow Block.

    This is to create Normalizing Flow (NF) Fast Flow model block based on
    Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions (List[int]): Input dimensions (Channel, Height, Width)
        conv3x3_only (bool): Boolean whether to use conv3x3 only or conv3x3 and conv1x1.
        hidden_ratio (float): Ratio for the hidden layer channels.
        flow_steps (int): Flow steps.
        clamp (float, optional): Clamp. Defaults to 2.0.

    Returns:
        SequenceINN: FastFlow Block.
    """
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = 1
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio, padding),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Cell):
    """FastFlow.

    Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

    Args:
        input_size (Tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        flow_steps (int, optional): Flow steps.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model. Defaults to False.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.

    Raises:
        ValueError: When the backbone is not supported.
    """

    def __init__(
            self,
            backbone,
            flow_steps,
            input_size,
            conv3x3_only=False,
            hidden_ratio=1.0
        ):
        super(FastFlow, self).__init__()

        self.feature_extractor = backbone
        channels = self.feature_extractor.channels
        scales = self.feature_extractor.reduction

        self.norms = nn.CellList()
        for in_channels, scale in zip(channels, scales):
            normalized_shape = [in_channels, int(input_size / scale), int(input_size / scale)]
            self.norms.append(
                nn.LayerNorm(normalized_shape, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-05)
            )

        self.nf_flows = nn.CellList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.init_weights(self.nf_flows)

    def init_weights(self, block):
        """
        Initialize network weights.

        Parameters:
            net (Cell): Network to be initialized
            init_type (str): The name of an initialization method: normal | xavier.
            init_gain (float): Gain factor for normal and xavier.

        """
        for _, cell in block.cells_and_names():
            if isinstance(cell, (nn.Conv2d)):
                cell.weight.set_data(init.initializer(init.HeUniform(math.sqrt(5)), cell.weight.shape))

    def construct(self, x):
        """Forward-Pass the input to the FastFlow Model.

        Args:
            x (Tensor): Input tensor.
        Returns:
            Union[Tuple[Tensor, Tensor], Tensor]: During training, return
                (hidden_variables, log-of-the-jacobian-determinants).
                During the validation/test, return the anomaly map.
        """
        features = self.feature_extractor(x)

        features_norm = []
        for i, feature in enumerate(features):
            feature_norm = self.norms[i](feature)
            features_norm.append(feature_norm)

        hidden_variables = []
        log_jacobians = []
        for i, feature in enumerate(features_norm):
            output, log_jac_dets = self.nf_flows[i](feature)
            hidden_variables.append(output)
            log_jacobians.append(log_jac_dets)

        return (hidden_variables, log_jacobians)

def build_model(backbone, flow_step=8, im_resize=256, conv3x3_only=False, hidden_ratio=1.0):
    model = FastFlow(
        backbone=backbone,
        flow_steps=flow_step,
        input_size=im_resize,
        conv3x3_only=conv3x3_only,
        hidden_ratio=hidden_ratio,
    )
    return model
