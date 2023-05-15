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
'''point head'''

import math
import numpy
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.ops import constexpr
from mindspore import context
from maskrcnn.model_utils.device_adapter import get_device_id

@constexpr
def get_tensor(data, datatype):
    return Tensor(input_data=data, dtype=datatype)

def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''kaiming normal'''
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return numpy.random.normal(0, std, size=inputs_shape).astype(numpy.float32)

def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    '''calculate correct fan'''
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


class StandardPointHead(nn.Cell):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, input_channels, num_classes):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        # todo 通道数改为81，无效label为0不变
        num_classes = num_classes
        fc_dim = 256
        num_fc = 3
        cls_agnostic_mask = False
        self.coarse_pred_each_layer = True
        input_channels = input_channels
        # fmt: on
        self.cat = ops.Concat(1)
        self.relu = ops.ReLU()

        fc_dim_in = input_channels + num_classes
        fc_layers = []
        for _ in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, pad_mode='valid', has_bias=True,
                           weight_init='he_normal')
            fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.seq = nn.SequentialCell(fc_layers)
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1,
                                   padding=0, pad_mode='valid', has_bias=True,
                                   weight_init='normal', bias_init='zeros')

    def construct(self, fine_grained_features, coarse_features):
        '''construct'''
        x = self.cat((fine_grained_features, coarse_features))
        for layer in self.seq.cell_list:
            x = self.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = self.cat((x, coarse_features))
        return self.predictor(x)


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=get_device_id())
    shape1 = (337, 256, 14)
    shape2 = (337, 81, 14)
    stdnormal = ops.StandardNormal(seed=2)
    x1 = stdnormal(shape1)
    x2 = stdnormal(shape2)
    net = StandardPointHead(input_channels=256, num_classes=81)
    out = net(x1, x2)
