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

"""ECONet"""
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore as ms
import mindspore.ops as ops
from mindspore.common import initializer as init
from src.utils import ConsensusModule


class Conv2dBlock(nn.Cell):
    """
    Conv2dBlock
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode='pad'):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, pad_mode=pad_mode, has_bias=has_bias)
        self.bn2d = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv2d(x)
        bn = self.bn2d(x)
        relu = self.relu(bn)
        return relu


class Inception(nn.Cell):
    """
    Inception Block
    """

    # self.block3b        256,       64,     64,   96,    64,      96,    96    64     )
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, n6x6, pool_planes):
        super(Inception, self).__init__()
        self.b1 = Conv2dBlock(in_channels, n1x1, kernel_size=1)
        self.b2 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red, kernel_size=1),
                                     Conv2dBlock(n3x3red, n3x3, kernel_size=3, padding=1)])
        self.b3 = nn.SequentialCell([Conv2dBlock(in_channels, n5x5red, kernel_size=1),
                                     Conv2dBlock(n5x5red, n5x5, kernel_size=3, padding=1),
                                     Conv2dBlock(n5x5, n6x6, kernel_size=3, padding=1)])
        self.pad_op = ops.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.b4 = Conv2dBlock(in_channels, pool_planes, kernel_size=1)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.b3(x)
        x = self.pad_op(x)
        cell = self.avgpool(x)
        branch4 = self.b4(cell)
        output = self.concat((branch1, branch2, branch3, branch4))
        return output


def conv3x3x3(channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same', has_bias=True):
    return nn.Conv3d(channels,
                     out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     pad_mode=pad_mode,
                     padding=padding,
                     has_bias=has_bias)


class _econet(nn.Cell):
    """
    backbone structure
    """

    def __init__(self, num_segments=4, include_top=True):
        super(_econet, self).__init__()
        self.count = 0
        self.num_segments = num_segments
        self.conv1 = Conv2dBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        self.conv2 = Conv2dBlock(64, 64, kernel_size=1)
        self.conv3 = Conv2dBlock(64, 192, kernel_size=3, padding=1)
        self.pad_op = ops.Pad(((0, 0), (0, 0), (0, 1), (0, 1)))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.block3a = Inception(192, 64, 64, 64, 64, 96, 96, 32)
        self.block3b = Inception(256, 64, 64, 96, 64, 96, 96, 64)

        self.conv4 = Conv2dBlock(320, 64, kernel_size=1)
        self.conv5 = Conv2dBlock(64, 96, kernel_size=3, padding=1)

        self.res3a_2_conv3d = conv3x3x3(96, 128, 3, 1, 0)
        self.res3a_bn_bn3d = nn.BatchNorm3d(128, momentum=0.9)
        self.relu = nn.ReLU()

        self.res3b_1_conv3d = conv3x3x3(128, 128, 3, 1, 0)
        self.res3b_1_bn_bn3d = nn.BatchNorm3d(128, momentum=0.9)

        self.res3b_2_conv3d = conv3x3x3(128, 128, 3, 1, 0)
        self.res3b_bn_bn3d = nn.BatchNorm3d(128, momentum=0.9)

        # **************************
        self.res4a_1_conv3d = conv3x3x3(128, 256, 3, 2, 1, pad_mode='pad')
        self.res4a_1_bn_bn3d = nn.BatchNorm3d(256, momentum=0.9)
        self.res4a_2_conv3d = conv3x3x3(256, 256, 3, 1, 0)
        self.res4a_down_conv3d = conv3x3x3(128, 256, 3, 2, 1, pad_mode='pad')
        # **************************
        self.res4a_bn_bn3d = nn.BatchNorm3d(256, momentum=0.9)
        self.res4b_1_conv3d = conv3x3x3(256, 256, 3, 1, 0)
        self.res4b_1_bn_bn3d = nn.BatchNorm3d(256, momentum=0.9)
        self.res4b_2_conv3d = conv3x3x3(256, 256, 3, 1, 0)
        # **************************
        self.res4b_bn_bn3d = nn.BatchNorm3d(256, momentum=0.9)
        self.res5a_1_conv3d = conv3x3x3(256, 512, 3, 2, 1, pad_mode='pad')
        self.res5a_1_bn_bn3d = nn.BatchNorm3d(512, momentum=0.9)
        self.res5a_2_conv3d = conv3x3x3(512, 512, 3, 1, 0)
        self.res5a_down_conv3d = conv3x3x3(256, 512, 3, 2, 1, pad_mode='pad')

        # ***************************
        self.res5a_bn_bn3d = nn.BatchNorm3d(512, momentum=0.9)
        self.res5b_1_conv3d = conv3x3x3(512, 512, 3, 1, 0)
        self.res5b_1_bn_bn3d = nn.BatchNorm3d(512, momentum=0.9)
        self.res5b_2_conv3d = conv3x3x3(512, 512, 3, 1, 0)

        self.res5b_bn_bn3d = nn.BatchNorm3d(512, momentum=0.9)
        self.kernel_size = int(self.num_segments / 4)
        self.reshape = ops.Reshape()
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.dense1 = nn.Dense(512, 400)

    def construct(self, x):
        """construct"""
        # 2D
        x = self.conv1(x)
        x = self.pad_op(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pad_op(x)
        x = self.maxpool2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # 2d -> 3d
        perm = (0, 2, 1, 3, 4)
        transpose = ms.ops.Transpose()
        out = transpose(x.view((-1, self.num_segments) + x.shape[1:]), perm)
        out = self.res3a_2_conv3d(out)
        res1 = out
        out = self.res3a_bn_bn3d(out)
        out = self.relu(out)
        out = self.res3b_1_conv3d(out)
        out = self.res3b_1_bn_bn3d(out)
        out = self.relu(out)
        out = self.res3b_2_conv3d(out)
        res2 = out
        out = res1 + res2
        out = self.res3b_bn_bn3d(out)
        out = self.relu(out)

        out1 = out
        out = self.res4a_1_conv3d(out)
        out = self.res4a_1_bn_bn3d(out)
        out = self.relu(out)  # (32, 256, 2, 14, 14)
        out = self.res4a_2_conv3d(out)
        res1 = out
        out = self.res4a_down_conv3d(out1)
        res2 = out
        out = res1 + res2

        res1 = out
        out = self.res4a_bn_bn3d(out)
        out = self.relu(out)
        out = self.res4b_1_conv3d(out)
        out = self.res4b_1_bn_bn3d(out)
        out = self.relu(out)
        out = self.res4b_2_conv3d(out)
        res2 = out
        out = res1 + res2

        out = self.res4b_bn_bn3d(out)
        out = self.relu(out)
        out1 = out
        out = self.res5a_1_conv3d(out)
        out = self.res5a_1_bn_bn3d(out)
        out = self.relu(out)
        out = self.res5a_2_conv3d(out)
        res1 = out
        out = self.res5a_down_conv3d(out1)
        res2 = out
        out = res1 + res2
        res1 = out
        out = self.res5a_bn_bn3d(out)
        out = self.relu(out)
        out = self.res5b_1_conv3d(out)
        out = self.res5b_1_bn_bn3d(out)
        out = self.relu(out)
        out = self.res5b_2_conv3d(out)
        res2 = out
        out = res1 + res2
        out = self.res5b_bn_bn3d(out)
        out = self.relu(out)
        out = self.reshape(out, (out.shape[0], out.shape[1], -1))
        out = self.reduce_mean(out, -1)
        out = self.dense1(out)
        return out


class ECONet(nn.Cell):
    """
    build ECO-lite net
    """

    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(ECONet, self).__init__()
        self.count = 0
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.num_class = num_class
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
        Initializing TSN with base model: {}.
        TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
                """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type,
                           self.dropout)))

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def construct(self, inputvar):
        """
        forward propagation
        """

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        inputvar = inputvar.view((-1, sample_len) + inputvar.shape[-2:])
        base_out = self.base_model(inputvar)
        if self.dropout < 1:
            base_out = self.new_fc(base_out)
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = self.consensus(base_out)
            return base_out
        return base_out

    def _prepare_tsn(self, num_class):
        """
        prepare tsn
        """
        feature_dim = self.base_model.dense1.in_channels

        if self.dropout == 1:
            self.base_model.dense1 = nn.Dense(feature_dim, num_class)
            self.new_fc = None
        else:
            self.base_model.dense1 = nn.Dropout(p=1 - self.dropout)

            self.new_fc = nn.Dense(feature_dim, num_class)

        if self.new_fc is None:
            shape = self.base_model.dense1.weight.shape
            self.base_model.dense1.weight.set_data(init.initializer('xavier_uniform', shape))
            shape = self.base_model.dense1.bias.shape
            self.base_model.dense1.bias.set_data(init.initializer('zeros', shape))

        else:
            shape = self.new_fc.weight.shape
            self.new_fc.weight.set_data(init.initializer('xavier_uniform', shape))
            shape = self.new_fc.bias.shape
            self.new_fc.bias.set_data(init.initializer('zeros', shape))
        return feature_dim

    def _prepare_base_model(self, base_model):
        self.base_model = _econet(self.num_segments)
        self.input_size = 224
        self.input_mean = [104, 117, 128]
        self.input_std = [1]

    def partialBN(self, enable):
        self._enable_pbn = enable

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_optim_policies(self):
        """get_optim_policies"""
        first_3d_conv_weight = []
        first_3d_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        bn_gamma_cnt = 0
        bn_beta_cnt = 0
        for x in self.trainable_params():
            parameter_name = x.name
            if parameter_name.endswith('conv2d.weight'):
                normal_weight.append(x)
            elif parameter_name.endswith('conv2d.bias'):
                normal_bias.append(x)
            elif parameter_name.endswith('res3a_2_conv3d.weight'):
                first_3d_conv_weight.append(x)
            elif parameter_name.endswith('res3a_2_conv3d.bias'):
                first_3d_conv_bias.append(x)
            elif parameter_name.endswith('conv3d.weight'):
                normal_weight.append(x)
            elif parameter_name.endswith('conv3d.bias'):
                normal_bias.append(x)
            elif parameter_name.endswith('bn2d.gamma'):
                bn_gamma_cnt += 1
                if not self._enable_pbn or bn_gamma_cnt == 1:
                    bn.append(x)
            elif parameter_name.endswith('bn2d.beta'):
                bn_beta_cnt += 1
                if not self._enable_pbn or bn_beta_cnt == 1:
                    bn.append(x)
            elif parameter_name.endswith('weight'):  # dense1.weight, new_fc.weight
                normal_weight.append(x)
            elif parameter_name.endswith('bias'):  # dense1.bias, new_fc.bias
                normal_bias.append(x)
        return [{'params': first_3d_conv_weight}, {'params': first_3d_conv_bias}, {'params': normal_weight},
                {'params': normal_bias}, {'params': bn}]
