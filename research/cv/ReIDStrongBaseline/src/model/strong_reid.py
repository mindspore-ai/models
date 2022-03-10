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
""" ReID Strong Baseline model """
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import HeNormal
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.model.resnet import resnet50


class ReIDStrong(nn.Cell):
    """ ReID Strong Baseline Model structure

    Args:
        num_classes: number of classes
        pretrained_backbone: path to resnet50 backbone
        training: flag for training mode
    """

    def __init__(self, num_classes=751, pretrained_backbone='', training=True):
        super().__init__()
        in_planes = 2048
        self.training = training

        resnet = resnet50()
        if pretrained_backbone:
            print(f'Load pretrain from {pretrained_backbone}')
            load_param_into_net(resnet, load_checkpoint(pretrained_backbone))

        self.backone = resnet

        self.gap = nn.AvgPool2d(kernel_size=(16, 8), stride=(16, 8))
        self.flat = ops.Flatten()
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.beta.requires_grad = False

        self.classifier = nn.Dense(in_planes, num_classes, has_bias=False,
                                   weight_init=HeNormal(negative_slope=0, mode="fan_out"))

    def construct(self, x):
        """ Forward """
        res_map = self.backone(x)
        global_feat = self.gap(res_map)

        global_feat = self.flat(global_feat)
        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        return feat
