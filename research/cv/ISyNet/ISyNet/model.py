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
"""ISyNet model"""
import json
from mindspore import nn
from .backbone import CustomBackbone
from .head import CustomHead
from .json_parser_backbone import get_layers

__all__ = ['ISyNet']

class ISyNet(nn.Cell):
    """ISyNet"""
    def __init__(self,
                 num_classes=1000,
                 json_arch_file_backbone='',
                 dropout=0.5,
                 weight_standardization=0,
                 last_bn=0,
                 dml=1,
                 evaluate=False):
        """ Network initialisation """
        super().__init__()
        self.dml = dml
        self.evaluate = evaluate
        if dml == 2:
            self.json_backbone_description = json_arch_file_backbone[0]
            self.weight_standardization = weight_standardization
            with open(self.json_backbone_description, "r", encoding="utf-8") as read_file:
                data_backbone = json.load(read_file)
                [backbone_layer_index, out_channels] = get_layers(data_backbone)

            self.backbone1 = CustomBackbone(data_backbone,
                                            backbone_layer_index,
                                            weight_standardization=self.weight_standardization)
            self.head1 = CustomHead(num_classes, out_channels, dropout, last_bn)
            print("model", json_arch_file_backbone[0], "created")
            self.json_backbone_description = json_arch_file_backbone[1]
            with open(self.json_backbone_description, "r", encoding="utf-8") as read_file:
                data_backbone = json.load(read_file)
                [backbone_layer_index, out_channels] = get_layers(data_backbone)

            self.backbone2 = CustomBackbone(data_backbone,
                                            backbone_layer_index,
                                            weight_standardization=self.weight_standardization)
            self.head2 = CustomHead(num_classes, out_channels, dropout, last_bn)
            print("model", json_arch_file_backbone[1], "created")

        else:
            self.json_backbone_description = json_arch_file_backbone
            self.weight_standardization = weight_standardization
            with open(self.json_backbone_description, "r", encoding="utf-8") as read_file:
                data_backbone = json.load(read_file)
                [backbone_layer_index, out_channels] = get_layers(data_backbone)

            self.backbone1 = CustomBackbone(data_backbone,
                                            backbone_layer_index,
                                            weight_standardization=self.weight_standardization)
            self.head1 = CustomHead(num_classes, out_channels, dropout, last_bn)

    def construct(self, *inputs, **_kwargs):
        """IsyNet construct for dml=1 and dml=2"""
        x = inputs[0]
        if self.dml > 1:
            y1 = self.backbone1(x)
            features1 = self.head1(y1)
            y2 = self.backbone2(x)
            features2 = self.head2(y2)
            if self.evaluate:
                features = features1
            else:
                features = [features1, features2]
        else:
            x = self.backbone1(x)
            features = self.head1(x)
        return features

    def init_weights(self, mode='fan_in', slope=0):
        """initialization of weights"""
        info_list = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                info_list.append(str(m))
                nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                info_list.append(str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                info_list.append(str(m))
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
