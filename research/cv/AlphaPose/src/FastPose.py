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
'''
Alphapose network
'''
import os
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net

from src.DUC import DUC
from src.SE_Resnet import SEResnet
from src.config import config


if config.MODELARTS_IS_MODEL_ARTS:
    pretrained = os.path.join(config.MODELARTS_CACHE_INPUT, config.MODEL_PRETRAINED)
else:
    pretrained = os.path.join(config.MODEL_PRETRAINED)


def createModel():
    '''
    createModel
    '''
    return FastPose_SE()

class FastPose_SE(nn.Cell):
    '''
    FastPose_SE
    '''
    conv_dim = 128
    def __init__(self):
        super(FastPose_SE, self).__init__()
        param_dict = load_checkpoint(pretrained)
        resnet50 = SEResnet('resnet50')
        load_param_into_net(resnet50, param_dict)
        self.preact = resnet50
        self.suffle1 = ops.DepthToSpace(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.conv_dim, config.TRAIN_nClasses, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
    def construct(self, x):
        '''
        construct
        '''
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)
        return out
