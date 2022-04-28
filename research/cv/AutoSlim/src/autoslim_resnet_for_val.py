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
"""
Autoslim resnet backbone for validation
"""
from mindspore import nn

from src.slimmable_ops import SwitchableBatchNorm2d
from src.slimmable_ops import SlimmableConv2d, SlimmableLinear
from src.slimmable_ops import pop_channels
from src.config import FLAGS

class Block(nn.Cell):
    def __init__(self, inp, outp, midp1, midp2, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        layers = [
            SlimmableConv2d(inp, midp1, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(midp1),
            nn.ReLU(),

            SlimmableConv2d(midp1, midp2, 3, stride, 1, bias=False),
            SwitchableBatchNorm2d(midp2),
            nn.ReLU(),

            SlimmableConv2d(midp2, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.SequentialCell(layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.SequentialCell(
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchableBatchNorm2d(outp),
            )
        self.post_relu = nn.ReLU()

    def construct(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res


class AutoSlimModel(nn.Cell):
    def __init__(self, num_classes=1000, input_size=224):
        super(AutoSlimModel, self).__init__()

        self.features = []
        assert input_size % 32 == 0

        channel_num_list = FLAGS.channel_num_list.copy()
        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]
        # feats = [64, 128, 256, 512]
        channels = pop_channels(FLAGS.channel_num_list)
        self.features.append(
            nn.SequentialCell(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], # [3,3,3,3]
                    channels, 7, 2, 3, bias=False),
                SwitchableBatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 'same'),
            )
        )

        # body
        for stage_id, n in enumerate(self.block_setting):
            for i in range(n):
                if i == 0:
                    outp = pop_channels(FLAGS.channel_num_list)
                midp1 = pop_channels(FLAGS.channel_num_list)
                midp2 = pop_channels(FLAGS.channel_num_list)
                outp = pop_channels(FLAGS.channel_num_list)
                if i == 0 and stage_id != 0:
                    self.features.append(
                        Block(channels, outp, midp1, midp2, 2))
                else:
                    self.features.append(
                        Block(channels, outp, midp1, midp2, 1))
                channels = outp

        # cifar10
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.SequentialCell(self.features)

        # classifier
        self.outp = channels
        FLAGS.channel_num_list = channel_num_list.copy()
        self.classifier = SlimmableLinear(
            self.outp,
            [num_classes for _ in range(len(self.outp))])

        if FLAGS.reset_parameters:
            self.init_parameters_data()

    def construct(self, x):
        # original network
        x = self.features(x)
        last_dim = x.shape[1]
        x = x.view(-1, last_dim)
        return self.classifier(x)
