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

from src.model.resnet import resnet


def _make_blocks(out_channel, stride, num_block):
    """
    make blocks config
    """
    result = []
    for ch, s, num in zip(out_channel, stride, num_block):
        result.append([{
            'out_channel': ch,
            'stride': 1
        }] * (num - 1) + [{
            'out_channel': ch,
            'stride': s
        }])
    return result


def resnet_101(in_channels, **kwargs):
    """
    get resnet 101
    """
    blocks = _make_blocks([256, 512, 1024, 2048], [2, 2, 2, 1], [3, 4, 23, 3])
    net = resnet.ResNet(blocks, in_channels, **kwargs)
    return net
