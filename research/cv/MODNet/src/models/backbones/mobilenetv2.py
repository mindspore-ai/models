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

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

def _make_divisible(v, divisor, min_value=None):
    "Useful functions for model"
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_bn(inp, oup, stride):
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, 3, stride, padding=1, pad_mode='pad', has_bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )

def conv_1x1_bn(inp, oup):
    return nn.SequentialCell([
        nn.Conv2d(inp, oup, 1, 1, padding=0, pad_mode='pad', has_bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()])

class InvertedResidual(nn.Cell):
    "Class of Inverted Residual block"
    def __init__(self, inp, oup, stride, expansion, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.SequentialCell(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1, group=hidden_dim,
                          dilation=dilation, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, padding=0, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.SequentialCell(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, padding=0, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1, group=hidden_dim,
                          dilation=dilation, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, padding=0, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(oup),
            )


    def construct(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)

        return out

class MobileNetV2(nn.Cell):
    "Class of MobileNetV2"
    def __init__(self, in_channels, alpha=1.0, expansion=6, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(input_channel * alpha, 8)
        self.last_channel = _make_divisible(last_channel * alpha, 8) if alpha > 1.0 else last_channel
        self.features = [conv_bn(self.in_channels, input_channel, 2)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = _make_divisible(int(c * alpha), 8)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features = nn.CellList(self.features)

        # Initialize weights
        self._init_weights()

    def construct(self, x):
        # Stage1
        x = self.features[0](x)
        x = self.features[1](x)
        # Stage2
        x = self.features[2](x)
        x = self.features[3](x)
        # Stage3
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        # Stage4
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        # Stage5
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)

        return x

    def _load_pretrained_model(self, pretrained_file):
        pretrain_dict = mindspore.load_checkpoint(pretrained_file)
        model_dict = {}
        state_dict = self.state_dict()
        print("[MobileNetV2] Loading pretrained model...")
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                print(k, "is ignored")
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _init_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
