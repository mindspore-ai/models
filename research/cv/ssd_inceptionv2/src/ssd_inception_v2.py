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

"""SSDFeatureExtractor for InceptionV2 features."""

import mindspore.nn  as nn
from mindspore.common.initializer import initializer, TruncatedNormal
from src import inception_v2
from src import ssd
from src import feature_map_generators


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if initialize_mode == 'TruncatedNormal':
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))
            else:
                p.set_data(initialize_mode, p.data.shape, p.data.dtype)


class SSD_inception_v2(nn.Cell):
    """SSD Feature Extractor using InceptionV2 features."""
    def __init__(self, configs, use_explicit_padding=False, depth_multiplier=1,
                 use_depthwise=False):
        super(SSD_inception_v2, self).__init__()
        self.lable_num = configs.num_classes
        self.use_explicit_padding = use_explicit_padding
        self.use_depthwise = use_depthwise
        self.box = ssd.MultiBox(configs)
        self.InceptionV2 = inception_v2.inception_v2_base()
        self.feature_map_channel = self.InceptionV2.feature_map_channels

        self.feature_map_layout = {
            'from_layer': ['Mixed_4c', 'Mixed_5c', '', '', '', ''],
            'layer_depth': [-1, -1, 512, 256, 256, 128],
            'use_explicit_padding': self.use_explicit_padding,
            'use_depthwise': self.use_depthwise,
        }

        self.feature_maps = feature_map_generators.MultiResolutionFeatureMaps(
            feature_map_layout=self.feature_map_layout,
            feature_map_channels=self.feature_map_channel,
            depth_multiplier=depth_multiplier
        )

    def preprocess(self, resized_inputs):
        """Scaling"""
        return (2.0 / 255.0) * resized_inputs - 1.0

    def construct(self, inputs):
        src = self.InceptionV2(inputs)
        src = self.feature_maps(src)
        locs, confs = self.box(src)
        return locs, confs
