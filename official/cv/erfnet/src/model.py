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
from mindspore import ops, nn
class DownsamplerBlock(nn.Cell):
    def __init__(self, in_feature_num, out_feature_num, weight_init):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(in_feature_num, out_feature_num-in_feature_num, \
            3, stride=2, padding=(1, 1, 1, 1), has_bias=True, \
            weight_init=weight_init, pad_mode="pad")
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_feature_num, eps=1e-3)
        self.cat = ops.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        a = self.conv(x)
        b = self.pool(x)
        output = self.cat((a, b))
        output = self.bn(output)
        output = self.relu(output)
        return output

class non_bottleneck_1d(nn.Cell):
    def __init__(self, chann, dropprob, dilated, weight_init):
        super(non_bottleneck_1d, self).__init__()
        self.dropprob = dropprob
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, \
            padding=(1, 1, 0, 0), pad_mode='pad', has_bias=True, \
            weight_init=weight_init)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, \
            padding=(0, 0, 1, 1), pad_mode='pad', has_bias=True, \
            weight_init=weight_init)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, \
            padding=(dilated, dilated, 0, 0), pad_mode='pad', \
            has_bias=True, dilation=(dilated, 1), weight_init=weight_init)
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, \
            padding=(0, 0, dilated, dilated), pad_mode='pad', \
            has_bias=True, dilation=(1, dilated), weight_init=weight_init)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        if dropprob > 0:
            self.dropout = ops.Dropout(keep_prob=1-dropprob)
        self.relu = nn.ReLU()
        self.dilated = dilated

    def construct(self, x):
        output = self.conv3x1_1(x)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropprob > 0:
            output, _ = self.dropout(output)
        return self.relu(output + x)

class Encoder(nn.Cell):
    def __init__(self, stage, weight_init, run_distribute, train=True):
        super(Encoder, self).__init__()

        if train:
            if run_distribute:
                if stage == 3:
                    drop_prob = [0.03, 0.2]
                elif stage in (1, 2, 4):
                    drop_prob = [0.03, 0.3]
                else:
                    raise RuntimeError("stage should be 1, 2, 3, or 4.")
            else:
                drop_prob = [0.03, 0.2]
        else:
            drop_prob = [.0, .0]

        self.layers = nn.CellList()
        self.down1 = DownsamplerBlock(3, 16, weight_init)
        self.down2 = DownsamplerBlock(16, 64, weight_init)

        self.bottleneck1 = non_bottleneck_1d(64, drop_prob[0], 1, weight_init)
        self.bottleneck2 = non_bottleneck_1d(64, drop_prob[0], 1, weight_init)
        self.bottleneck3 = non_bottleneck_1d(64, drop_prob[0], 1, weight_init)
        self.bottleneck4 = non_bottleneck_1d(64, drop_prob[0], 1, weight_init)
        self.bottleneck5 = non_bottleneck_1d(64, drop_prob[0], 1, weight_init)

        self.down3 = DownsamplerBlock(64, 128, weight_init)

        self.bottleneck6 = non_bottleneck_1d(128, drop_prob[1], 2, weight_init)
        self.bottleneck7 = non_bottleneck_1d(128, drop_prob[1], 4, weight_init)
        self.bottleneck8 = non_bottleneck_1d(128, drop_prob[1], 8, weight_init)
        self.bottleneck9 = non_bottleneck_1d(128, drop_prob[1], 16, weight_init)

        self.bottleneck10 = non_bottleneck_1d(128, drop_prob[1], 2, weight_init)
        self.bottleneck11 = non_bottleneck_1d(128, drop_prob[1], 4, weight_init)
        self.bottleneck12 = non_bottleneck_1d(128, drop_prob[1], 8, weight_init)
        self.bottleneck13 = non_bottleneck_1d(128, drop_prob[1], 16, weight_init)

    def construct(self, x):
        x = self.down1(x)
        x = self.down2(x)

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)

        x = self.down3(x)

        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        return x

class UpsamplerBlock(nn.Cell):
    def __init__(self, in_feature_num, out_feature_num, weight_init):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.Conv2dTranspose(in_feature_num, out_feature_num, 3, \
            stride=2, has_bias=True, weight_init=weight_init)
        self.bn = nn.BatchNorm2d(out_feature_num, eps=1e-03)
        self.relu = nn.ReLU()
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return  x

class Decoder(nn.Cell):
    def __init__(self, num_classes, weight_init):
        super(Decoder, self).__init__()

        self.up1 = UpsamplerBlock(128, 64, weight_init)
        self.bottleneck1 = non_bottleneck_1d(64, 0, 1, weight_init)
        self.bottleneck2 = non_bottleneck_1d(64, 0, 1, weight_init)

        self.up2 = UpsamplerBlock(64, 16, weight_init)
        self.bottleneck3 = non_bottleneck_1d(16, 0, 1, weight_init)
        self.bottleneck4 = non_bottleneck_1d(16, 0, 1, weight_init)

        self.pred = nn.Conv2dTranspose(16, num_classes, 2, stride=2, has_bias=True, \
                           weight_init=weight_init)

    def construct(self, x):
        x = self.up1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        x = self.up2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.pred(x)
        return x

class Encoder_pred(nn.Cell):
    def __init__(self, stage, num_class, weight_init, run_distribute, train=True):
        super(Encoder_pred, self).__init__()
        self.encoder = Encoder(stage, weight_init, run_distribute, train)
        self.pred = nn.Conv2d(128, num_class, 1, stride=1, pad_mode='valid', \
                                  has_bias=True, weight_init=weight_init)
    def construct(self, x):
        x = self.encoder(x)
        x = self.pred(x)
        return x

class ERFNet(nn.Cell):
    def __init__(self, stage, num_class, init_conv, run_distribute, train=True):
        super(ERFNet, self).__init__()
        self.encoder = Encoder(stage, init_conv, run_distribute, train)
        self.decoder = Decoder(num_class, init_conv)
    def construct(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
