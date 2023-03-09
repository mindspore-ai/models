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
""" effnet model """
from src.efficientnet.utils import(
    get_model_params,
    Swish,
    MemoryEfficientSwish,
)
import mindspore
from mindspore import nn
import mindspore.ops as op
from mindspore.common import set_seed
set_seed(1)


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1. - p
    random_tensor = keep_prob
    shape = (batch_size, 1, 1, 1)
    uniformreal = op.UniformReal(0)
    random_tensor += uniformreal(shape)
    floor = mindspore.ops.Floor()
    binary_tensor = floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Cell):
    """ MBConv Block"""
    def __init__(self, block_args, global_params, drop_rate=0.0, is_training=False):
        super().__init__()
        self.is_training = is_training
        self._block_args = block_args
        self._bn_mom = 1 - 0.01
        self._bn_eps = 1e-3
        self.has_se = (self._block_args["se_ratio"] is not None) and (0 < self._block_args["se_ratio"] <= 1)

        self.drop_rate = drop_rate
        if drop_rate != 0:
            self.drop_layer = nn.Dropout(p=drop_rate)
        self.id_skip = True
        inp = self._block_args["input_filters"]  # number of input channels
        oup = self._block_args["input_filters"] * self._block_args["expand_ratio"]  # number of output channels
        if self._block_args["expand_ratio"] != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, pad_mode="same",
                                          padding=0, has_bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, eps=self._bn_eps, momentum=0.99)

        k = self._block_args["kernel_size"]
        s = self._block_args["stride"]

        self._depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=k, stride=s, pad_mode="same",
                                         padding=0, group=oup, weight_init="he_uniform", has_bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, eps=1e-3, momentum=0.99)

        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args["input_filters"] * self._block_args["se_ratio"]))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, pad_mode="same",
                                        padding=0, kernel_size=1, weight_init="he_uniform", has_bias=True)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, pad_mode="same",
                                        padding=0, kernel_size=1, weight_init="he_uniform", has_bias=True)

        final_oup = self._block_args["output_filters"]
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, stride=1,
                                       pad_mode="same", padding=0, weight_init="he_uniform", has_bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, eps=self._bn_eps, momentum=0.99)
        self._swish = MemoryEfficientSwish()
        self.reduce_op = op.ReduceMean(keep_dims=True)
        self.sig = nn.Sigmoid()
        self.sum = op.ReduceSum(False)
        self.act_fn = op.ReLU()
        self.gate_fn = op.Sigmoid()

    def construct(self, inputs):
        """ MBConv Block forward """
        x = inputs
        if self._block_args["expand_ratio"] != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.has_se:
            x_squeezed = self.reduce_op(x, (2, 3))
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = self.gate_fn(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        input_filters, output_filters = self._block_args["input_filters"], self._block_args["output_filters"]
        if self.id_skip and self._block_args["stride"] == 1 and input_filters == output_filters:
            if self.drop_rate:
                x = drop_connect(x, self.drop_rate, self.is_training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Cell):
    """ effnet """
    def __init__(self, blocks_args=None, global_params=None, is_training=False):
        super().__init__()
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.bn_mom = 0.01
        bn_eps = 1e-3

        self.conv_stem = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2,
                                   pad_mode="same", has_bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=0.99)
        self.blocks = nn.CellList([])

        for idx, block_args in enumerate(self._blocks_args):
            self.blocks.append(MBConvBlock(block_args, self._global_params, drop_rate=0.2 * idx / 16,
                                           is_training=is_training))

        self.swish = MemoryEfficientSwish()

        # classifier part. not used
        self.mean = op.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.conv_head = nn.Conv2d(320, 1280, 1, 1, pad_mode="same")
        self.end_point = nn.Dense(1280, 1000)

        self._bn2 = nn.BatchNorm2d(num_features=1280, eps=bn_eps, momentum=0.99)
        self.relu = op.ReLU()

    @classmethod
    def from_name(cls, model_name, is_training=False, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, is_training)

    @classmethod
    def from_pretrained(cls, model_name, load_weights=False, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    def construct(self, inputs):
        """ effnet forward """

        x = self.conv_stem(inputs)   # Stem

        x = self.bn0(x)
        x = self.swish(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_head(x)
        x = self._bn2(x)
        x = self.relu(x)

        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        x = self.end_point(x)

        return x
