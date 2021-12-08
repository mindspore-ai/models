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
# ===========================================================================
"""Network of Auto-DeepLab"""
import numpy as np

import mindspore.nn as nn

from .aspp import ASPP
from .decoder import Decoder
from .encoder import get_default_arch, Encoder


class AutoDeepLab(nn.Cell):
    """Auto-DeepLab"""
    def __init__(self, args=None):
        super(AutoDeepLab, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        if args.net_arch is not None and args.cell_arch is not None:
            network_arch, cell_arch = np.load(args.net_arch), np.load(args.cell_arch)
        else:
            network_arch, cell_arch = get_default_arch()
        num_of_layers = len(network_arch)

        self.encoder = Encoder(network_arch,
                               cell_arch,
                               num_of_layers,
                               args.filter_multiplier,
                               args.block_multiplier,
                               args=args)

        self.aspp = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[network_arch[-1]],
                         momentum=args.bn_momentum,
                         eps=args.bn_eps,
                         parallel=args.parallel)

        self.decoder = Decoder(args.num_classes,
                               args.filter_multiplier * args.block_multiplier * filter_param_dict[network_arch[2]],
                               args.bn_momentum,
                               args.bn_eps,
                               parallel=args.parallel)

        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, x):
        """construct"""
        encoder_output, low_level_feature = self.encoder(x)
        high_level_feature = self.aspp(encoder_output)
        decoder_output = self.decoder(high_level_feature, low_level_feature)
        output = self.resize_bilinear(decoder_output, (x.shape[2], x.shape[3]), None, True)
        return output
