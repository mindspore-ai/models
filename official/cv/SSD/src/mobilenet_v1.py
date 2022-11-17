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

import mindspore.nn as nn
import mindspore.ops as ops

def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="same",
                            group=1 if not depthwise else in_channel))
    output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class MobileNetV1(nn.Cell):
    """
    MobileNet V1 backbone
    """
    def __init__(self, class_num=1001, features_only=False):
        super(MobileNetV1, self).__init__()
        self.features_only = features_only
        cnn = [
            conv_bn_relu(3, 32, 3, 2, False),      # Conv0

            conv_bn_relu(32, 32, 3, 1, True),      # Conv1_depthwise
            conv_bn_relu(32, 64, 1, 1, False),     # Conv1_pointwise
            conv_bn_relu(64, 64, 3, 2, True),      # Conv2_depthwise
            conv_bn_relu(64, 128, 1, 1, False),    # Conv2_pointwise

            conv_bn_relu(128, 128, 3, 1, True),    # Conv3_depthwise
            conv_bn_relu(128, 128, 1, 1, False),   # Conv3_pointwise
            conv_bn_relu(128, 128, 3, 2, True),    # Conv4_depthwise
            conv_bn_relu(128, 256, 1, 1, False),   # Conv4_pointwise

            conv_bn_relu(256, 256, 3, 1, True),    # Conv5_depthwise
            conv_bn_relu(256, 256, 1, 1, False),   # Conv5_pointwise
            conv_bn_relu(256, 256, 3, 2, True),    # Conv6_depthwise
            conv_bn_relu(256, 512, 1, 1, False),   # Conv6_pointwise

            conv_bn_relu(512, 512, 3, 1, True),    # Conv7_depthwise
            conv_bn_relu(512, 512, 1, 1, False),   # Conv7_pointwise
            conv_bn_relu(512, 512, 3, 1, True),    # Conv8_depthwise
            conv_bn_relu(512, 512, 1, 1, False),   # Conv8_pointwise
            conv_bn_relu(512, 512, 3, 1, True),    # Conv9_depthwise
            conv_bn_relu(512, 512, 1, 1, False),   # Conv9_pointwise
            conv_bn_relu(512, 512, 3, 1, True),    # Conv10_depthwise
            conv_bn_relu(512, 512, 1, 1, False),   # Conv10_pointwise
            conv_bn_relu(512, 512, 3, 1, True),    # Conv11_depthwise
            conv_bn_relu(512, 512, 1, 1, False),   # Conv11_pointwise

            conv_bn_relu(512, 512, 3, 2, True),    # Conv12_depthwise
            conv_bn_relu(512, 1024, 1, 1, False),  # Conv12_pointwise
            conv_bn_relu(1024, 1024, 3, 1, True),  # Conv13_depthwise
            conv_bn_relu(1024, 1024, 1, 1, False), # Conv13_pointwise
        ]

        if self.features_only:
            self.network = nn.CellList(cnn)
        else:
            self.network = nn.SequentialCell(cnn)
            self.fc = nn.Dense(1024, class_num)

    def construct(self, x):
        output = x
        if self.features_only:
            features = ()
            for block in self.network:
                output = block(output)
                features = features + (output,)
            return features
        output = self.network(x)
        output = ops.ReduceMean()(output, (2, 3))
        output = self.fc(output)
        return output

class FeatureSelector(nn.Cell):
    """
    Select specific layers from an entire feature list
    """
    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def construct(self, feature_list):
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected

class MobileNetV1Feature(nn.Cell):
    """
    MobileNetV1 with FPN as SSD backbone.
    """
    def __init__(self, config):
        super(MobileNetV1Feature, self).__init__()
        self.mobilenet_v1 = MobileNetV1(features_only=True)

        self.selector = FeatureSelector([14, 26])

        self.layer_indexs = [14, 26]

    def construct(self, x):
        features = self.mobilenet_v1(x)
        features = self.selector(features)
        return features

def mobilenet_v1(class_num=1001):
    return MobileNetV1(class_num)

def mobilenet_v1_Feature(config):
    return MobileNetV1Feature(config)
