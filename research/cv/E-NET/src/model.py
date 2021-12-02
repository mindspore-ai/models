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
"""enet model"""
from mindspore import nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P


class InitialBlock(nn.Cell):
    """initial block"""
    def __init__(self, in_channels, out_channels, weight_init, relu):
        super(InitialBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            pad_mode='pad',
            padding=1,
            has_bias=False,
            weight_init=weight_init
        )
        if relu:
            self.out_activation = nn.ReLU()
        else:
            self.out_activation = nn.PReLU(out_channels)

        self.ext_branch = nn.MaxPool2d(3, stride=2, pad_mode='same')

        self.concat = P.Concat(axis=1)

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        """construct"""
        main = self.conv(x)

        side = self.ext_branch(x)

        # concatenating on the channels axis
        cat = self.concat((main, side))

        cat = self.batch_norm(cat)
        out = self.out_activation(cat)

        return out


class RegularBottleNeck(nn.Cell):
    """regular bottleneck"""
    def __init__(self, channels, drop_prob, weight_init, relu):
        super(RegularBottleNeck, self).__init__()

        self.drop_prob = drop_prob
        self.reduced_depth = int(channels // 4)
        if drop_prob > 0:
            self.dropout = P.Dropout(keep_prob=1-drop_prob)

        self.concat = P.Concat(axis=1)

        self.conv1 = nn.SequentialCell(nn.Conv2d(in_channels=channels,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=1,
                                                 stride=1,
                                                 pad_mode='valid',
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),

                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.conv2 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=3,
                                                 stride=1,
                                                 pad_mode='pad',
                                                 padding=1,
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.conv3 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 pad_mode='valid',
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(channels),
                                       nn.ReLU() if relu else nn.PReLU(channels))

        self.out_activation = nn.ReLU() if relu else nn.PReLU(channels)

    def construct(self, x):
        """construct"""
        x_copy = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.drop_prob > 0:
            x, _ = self.dropout(x)
        x = self.out_activation(x + x_copy)

        return x


class DlilatedBottleNeck(nn.Cell):
    """bottleneck with dlilated convolution"""
    def __init__(self, channels, drop_prob, dilation, weight_init, relu):
        super(DlilatedBottleNeck, self).__init__()

        self.drop_prob = drop_prob
        self.dilation = dilation
        self.reduced_depth = int(channels // 4)
        if drop_prob > 0:
            self.dropout = P.Dropout(keep_prob=1-drop_prob)
        self.concat = P.Concat(axis=1)

        self.conv1 = nn.SequentialCell(nn.Conv2d(in_channels=channels,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=1,
                                                 stride=1,
                                                 pad_mode='valid',
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),

                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.conv2 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=3,
                                                 stride=1,
                                                 pad_mode='pad',
                                                 padding=self.dilation,
                                                 dilation=self.dilation,
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.conv3 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 pad_mode='valid',
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(channels),
                                       nn.ReLU() if relu else nn.PReLU(channels))

        self.out_activation = nn.ReLU() if relu else nn.PReLU(channels)

    def construct(self, x):
        """construct"""
        x_copy = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.drop_prob > 0:
            x, _ = self.dropout(x)
        x = self.out_activation(x + x_copy)

        return x


class Asym5BottleNeck(nn.Cell):
    """bottleneck with a convolution"""
    def __init__(self, channels, drop_prob, weight_init, relu):
        super(Asym5BottleNeck, self).__init__()

        self.drop_prob = drop_prob
        self.reduced_depth = int(channels // 4)
        if drop_prob > 0:
            self.dropout = P.Dropout(keep_prob=1-drop_prob)
        self.concat = P.Concat(axis=1)

        self.conv1 = nn.SequentialCell(nn.Conv2d(in_channels=channels,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=1,
                                                 stride=1,
                                                 pad_mode='valid',
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),

                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.conv2 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=(1, 5),
                                                 stride=1,
                                                 pad_mode='pad',
                                                 padding=(0, 0, 2, 2),
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth),

                                       nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=self.reduced_depth,
                                                 kernel_size=(5, 1),
                                                 stride=1,
                                                 pad_mode='pad',
                                                 padding=(2, 2, 0, 0),
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(self.reduced_depth),
                                       nn.ReLU() if relu else nn.PReLU(self.reduced_depth))
        self.conv3 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                 out_channels=channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 pad_mode='valid',
                                                 dilation=1,
                                                 has_bias=False,
                                                 weight_init=weight_init),
                                       nn.BatchNorm2d(channels),
                                       nn.ReLU() if relu else nn.PReLU(channels))

        self.out_activation = nn.ReLU() if relu else nn.PReLU(channels)

    def construct(self, x):
        """construct"""
        x_copy = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.drop_prob > 0:
            x, _ = self.dropout(x)
        x = self.out_activation(x + x_copy)

        return x


class DownsampleNeck(nn.Cell):
    """DownsampleNeck"""
    def __init__(self, in_channels, out_channels, drop_prob, weight_init, relu):
        super(DownsampleNeck, self).__init__()

        self.drop_prob = drop_prob
        self.reduced_depth = int(in_channels // 4)

        self.main_max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.concat = P.Concat(axis=1)

        self.ext_conv1 = nn.SequentialCell(nn.Conv2d(in_channels=in_channels,
                                                     out_channels=self.reduced_depth,
                                                     kernel_size=2,
                                                     stride=2,
                                                     pad_mode='valid',
                                                     has_bias=False,
                                                     weight_init=weight_init),
                                           nn.BatchNorm2d(self.reduced_depth),
                                           nn.ReLU() if relu else nn.PReLU(self.reduced_depth)
                                           )

        self.ext_conv2 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                     out_channels=self.reduced_depth,
                                                     kernel_size=3,
                                                     stride=1,
                                                     pad_mode='pad',
                                                     padding=1,
                                                     weight_init=weight_init),
                                           nn.BatchNorm2d(self.reduced_depth),
                                           nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.ext_conv3 = nn.SequentialCell(nn.Conv2d(in_channels=self.reduced_depth,
                                                     out_channels=out_channels,
                                                     kernel_size=1,
                                                     stride=1, has_bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU() if relu else nn.PReLU(out_channels))

        if drop_prob > 0:
            self.ext_regul = P.Dropout(1-drop_prob)

        self.out_activation = nn.ReLU() if relu else nn.PReLU(out_channels)

    def construct(self, x):
        """construct"""
        main = self.main_max1(x)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        if self.drop_prob > 0:
            ext, _ = self.ext_regul(ext)

        b, ch_ext, h, w = F.shape(ext)
        ch_main = F.shape(main)[1]

        zeros = P.Zeros()
        padding = zeros((b, ch_ext - ch_main, h, w), ext.dtype)
        main = self.concat((main, padding))
        out = main + ext

        return self.out_activation(out)


class UpsampleNeck(nn.Cell):
    """UpsampleNeck"""
    def __init__(self, in_channels, out_channels, drop_prob, weight_init, relu):
        super(UpsampleNeck, self).__init__()

        self.reduced_depth = int(in_channels // 4)

        self.main_conv1 = nn.SequentialCell(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                      has_bias=False, weight_init=weight_init),
                                            nn.BatchNorm2d(out_channels))

        self.ext_conv1 = nn.SequentialCell(nn.Conv2d(in_channels=in_channels, out_channels=self.reduced_depth,
                                                     kernel_size=1, has_bias=False, weight_init=weight_init),
                                           nn.BatchNorm2d(self.reduced_depth),
                                           nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.ext_tconv1 = nn.SequentialCell(nn.Conv2dTranspose(in_channels=self.reduced_depth,
                                                               out_channels=self.reduced_depth,
                                                               kernel_size=2,
                                                               stride=2,
                                                               pad_mode='valid',
                                                               weight_init=weight_init),
                                            nn.BatchNorm2d(self.reduced_depth),
                                            nn.ReLU() if relu else nn.PReLU(self.reduced_depth))

        self.ext_conv2 = nn.SequentialCell(nn.Conv2d(self.reduced_depth, out_channels, kernel_size=1, has_bias=False),
                                           nn.BatchNorm2d(out_channels))

        if drop_prob > 0:
            self.ext_regul = P.Dropout(1-drop_prob)
        self.drop_prob = drop_prob
        self.out_activation = nn.ReLU() if relu else nn.PReLU(out_channels)

    def construct(self, x):
        """construct"""
        main = self.main_conv1(x)
        _, _, h, w = F.shape(main)
        main = P.ResizeBilinear((h * 2, w * 2))(main)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext)
        ext = self.ext_conv2(ext)
        if self.drop_prob > 0:
            ext, _ = self.ext_regul(ext)

        out = main + ext
        return self.out_activation(out)


class Encoder(nn.Cell):
    """Encoder"""
    def __init__(self, weight_init, relu=False, train=True):
        super(Encoder, self).__init__()

        if train:
            drop_prob = [0.01, 0.1]
        else:
            drop_prob = [.0, .0]
        # The initial block
        self.initial_block = InitialBlock(3, 16, weight_init, relu)
        # The first bottleneck
        self.downsample1_0 = DownsampleNeck(16, 64, drop_prob[0], weight_init, relu)
        self.regular1_1 = RegularBottleNeck(64, drop_prob[0], weight_init, relu)
        self.regular1_2 = RegularBottleNeck(64, drop_prob[0], weight_init, relu)
        self.regular1_3 = RegularBottleNeck(64, drop_prob[0], weight_init, relu)
        self.regular1_4 = RegularBottleNeck(64, drop_prob[0], weight_init, relu)
        # The second bottleneck
        self.downsample2_0 = DownsampleNeck(64, 128, drop_prob[1], weight_init, relu)
        self.regular2_1 = RegularBottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated2_2 = DlilatedBottleNeck(128, drop_prob[1], 2, weight_init, relu)
        self.asymmetric2_3 = Asym5BottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated2_4 = DlilatedBottleNeck(128, drop_prob[1], 4, weight_init, relu)
        self.regular2_5 = RegularBottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated2_6 = DlilatedBottleNeck(128, drop_prob[1], 8, weight_init, relu)
        self.asymmetric2_7 = Asym5BottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated2_8 = DlilatedBottleNeck(128, drop_prob[1], 16, weight_init, relu)
        # The third bottleneck
        self.regular3_0 = RegularBottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated3_1 = DlilatedBottleNeck(128, drop_prob[1], 2, weight_init, relu)
        self.asymmetric3_2 = Asym5BottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated3_3 = DlilatedBottleNeck(128, drop_prob[1], 4, weight_init, relu)
        self.regular3_4 = RegularBottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated3_5 = DlilatedBottleNeck(128, drop_prob[1], 8, weight_init, relu)
        self.asymmetric3_6 = Asym5BottleNeck(128, drop_prob[1], weight_init, relu)
        self.dilated3_7 = DlilatedBottleNeck(128, drop_prob[1], 16, weight_init, relu)

    def construct(self, x):
        """construct"""
        # The initial block
        x = self.initial_block(x)

        # The first bottleneck
        x = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # The second bottleneck
        x = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # The third bottleneck
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        return x


class Decoder(nn.Cell):
    """Decoder"""
    def __init__(self, num_classes, weight_init, relu=True, train=True):
        super(Decoder, self).__init__()

        if train:
            drop_prob = 0.1
        else:
            drop_prob = .0
        self.upsample4_0 = UpsampleNeck(128, 64, drop_prob, weight_init, relu)
        self.regular4_1 = RegularBottleNeck(64, drop_prob, weight_init, relu)
        self.regular4_2 = RegularBottleNeck(64, drop_prob, weight_init, relu)
        self.upsample5_0 = UpsampleNeck(64, 16, drop_prob, weight_init, relu)
        self.regular5_1 = RegularBottleNeck(16, drop_prob, weight_init, relu)
        self.fullconv = nn.Conv2dTranspose(16,
                                           num_classes,
                                           kernel_size=3,
                                           stride=2,
                                           pad_mode="same",
                                           weight_init=weight_init, has_bias=False)

    def construct(self, x):
        """construct"""
        x = self.upsample4_0(x)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x)
        x = self.regular5_1(x)
        x = self.fullconv(x)

        return x


class Encoder_pred(nn.Cell):
    """Encoder+pred"""
    def __init__(self, num_class, weight_init, train=True):
        super(Encoder_pred, self).__init__()
        self.encoder = Encoder(
            weight_init, train=train, relu=False)
        self.pred = nn.Conv2d(128, num_class, kernel_size=1, stride=1,
                              pad_mode='valid', has_bias=True, weight_init=weight_init)

    def construct(self, x):
        """construct"""
        x = self.encoder(x)
        x = self.pred(x)
        return x


class Enet(nn.Cell):
    """complete Enet"""
    def __init__(self, num_classes, init_conv, train=True):
        super(Enet, self).__init__()
        self.encoder = Encoder(init_conv, train=train, relu=False)
        self.decoder = Decoder(num_classes, init_conv,
                               train=train, relu=True)

    def construct(self, x):
        """construct"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
