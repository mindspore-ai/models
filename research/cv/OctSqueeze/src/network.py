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

import mindspore.nn as nn
import mindspore.ops as ops


def network_creator(layer_list, in_channels):
    layers = []
    in_channels = in_channels
    for v in layer_list:
        mlp_layer = nn.Dense(in_channels, v, activation=None)
        layers += [mlp_layer, nn.ReLU()]
        in_channels = v
    return layers


class OctSqueezeNet(nn.Cell):
    def __init__(self):
        super().__init__()

        self.feature_layers = nn.CellList(network_creator([128, 128, 128, 128, 128], 6))
        self.aggregation_layers1 = nn.CellList(network_creator([128, 128, 128], 128 * 2))
        self.aggregation_layers2 = nn.CellList(network_creator([128, 128, 128], 128 * 2))
        self.softmax = nn.Softmax()
        self.cat = ops.Concat(axis=1)
        self.last_linear = nn.Dense(256, 256, activation=None)

    def construct(self, data):
        cur_node = data[:, :6]
        parent_1 = data[:, 6:12]
        parent_2 = data[:, 12:18]
        parent_3 = data[:, 18:24]
        for k in range(len(self.feature_layers)):
            cur_node = self.feature_layers[k](cur_node)
        for k in range(len(self.feature_layers)):
            parent_1 = self.feature_layers[k](parent_1)
        for k in range(len(self.feature_layers)):
            parent_2 = self.feature_layers[k](parent_2)
        for k in range(len(self.feature_layers)):
            parent_3 = self.feature_layers[k](parent_3)

        aggregation_c_p1 = self.cat((cur_node, parent_1))
        aggregation_c_p1 = self.aggregation_layers1[0](aggregation_c_p1)
        aggregation_c_p1 = self.aggregation_layers1[1](aggregation_c_p1)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_c_p1 = aggregation_c_p1 + self.aggregation_layers1[k](aggregation_c_p1)
            aggregation_c_p1 = self.aggregation_layers1[k + 1](aggregation_c_p1)

        aggregation_p1_p2 = self.cat((parent_1, parent_2))
        aggregation_p1_p2 = self.aggregation_layers1[0](aggregation_p1_p2)
        aggregation_p1_p2 = self.aggregation_layers1[1](aggregation_p1_p2)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_p1_p2 = aggregation_p1_p2 + self.aggregation_layers1[k](aggregation_p1_p2)
            aggregation_p1_p2 = self.aggregation_layers1[k + 1](aggregation_p1_p2)

        aggregation_c_p1_p2 = self.cat((aggregation_c_p1, aggregation_p1_p2))
        aggregation_c_p1_p2 = self.aggregation_layers2[0](aggregation_c_p1_p2)
        aggregation_c_p1_p2 = self.aggregation_layers2[1](aggregation_c_p1_p2)
        for k in range(2, len(self.aggregation_layers2), 2):
            aggregation_c_p1_p2 = aggregation_c_p1_p2 + self.aggregation_layers2[k](aggregation_c_p1_p2)
            aggregation_c_p1_p2 = self.aggregation_layers2[k + 1](aggregation_c_p1_p2)

        aggregation_p2_p3 = self.cat((parent_2, parent_3))
        aggregation_p2_p3 = self.aggregation_layers1[0](aggregation_p2_p3)
        aggregation_p2_p3 = self.aggregation_layers1[1](aggregation_p2_p3)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_p2_p3 = aggregation_p2_p3 + self.aggregation_layers1[k](aggregation_p2_p3)
            aggregation_p2_p3 = self.aggregation_layers1[k + 1](aggregation_p2_p3)

        aggregation_p1_p2_p3 = self.cat((aggregation_p1_p2, aggregation_p2_p3))
        aggregation_p1_p2_p3 = self.aggregation_layers2[0](aggregation_p1_p2_p3)
        aggregation_p1_p2_p3 = self.aggregation_layers2[1](aggregation_p1_p2_p3)
        for k in range(2, len(self.aggregation_layers2), 2):
            aggregation_p1_p2_p3 = aggregation_p1_p2_p3 + self.aggregation_layers2[k](aggregation_p1_p2_p3)
            aggregation_p1_p2_p3 = self.aggregation_layers2[k + 1](aggregation_p1_p2_p3)

        aggregation_c_p1_p2_p3 = self.cat((aggregation_c_p1_p2, aggregation_p1_p2_p3))

        feature = aggregation_c_p1_p2_p3.squeeze()
        out = self.last_linear(feature)

        return out


if __name__ == "__main__":
    # For test, print parts of model
    model = OctSqueezeNet()
    print(model.aggregation_layers1[0])
