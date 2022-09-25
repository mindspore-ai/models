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
import os
import mindspore
import mindspore.nn as nn
from src.models.backbones.mobilenetv2 import MobileNetV2


class BaseBackbone(nn.Cell):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def construct(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def construct(self, x):
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model is provided by github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = r'./pretrained/mindspore_mobilenetv2_human_seg.ckpt'

        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            print(os.path.exists(ckpt_path))
            exit()

        param = mindspore.load_checkpoint(ckpt_path)
        param = {'lr_branch.backbone.model.'+key: value for (key, value) in param.items()}
        mindspore.load_param_into_net(self.model, param)
        print('load pretrained mobilenetv2 backbone')

if __name__ == '__main__':
    net = MobileNetV2Backbone(3)
    net.load_pretrained_ckpt()
