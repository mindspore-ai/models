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
""" MGN model """
import copy

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import resnet50, ResidualBlock


class MGN(nn.Cell):
    """ Multiple Granularity Network model

    Args:
        num_classes: number of classes
        feats: number of output features
        pool: pooling type: avg|max
        pretrained_backbone: path to pretrained resnet50 backbone
    """
    def __init__(self, num_classes=751, feats=256, pool='avg', pretrained_backbone=''):
        super().__init__()

        resnet = resnet50()
        if pretrained_backbone:
            load_param_into_net(resnet, load_checkpoint(pretrained_backbone))

        self.backone = nn.SequentialCell(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(),
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.SequentialCell(*resnet.layer3[1:])
        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.SequentialCell(
            ResidualBlock(1024, 2048,
                          down_sample_layer=nn.SequentialCell(
                              nn.Conv2d(1024, 2048, 1, has_bias=False),
                              nn.BatchNorm2d(2048)),
                          ),
            ResidualBlock(2048, 2048),
            ResidualBlock(2048, 2048))

        lr4_params = resnet.layer4.parameters_dict()
        lr4_params = {k.split('.', 1)[-1]: v for k, v in lr4_params.items()}

        load_param_into_net(res_p_conv5, lr4_params)

        self.p1 = nn.SequentialCell(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.SequentialCell(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.SequentialCell(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        if pool == 'max':
            pool2d = nn.MaxPool2d
        elif pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise ValueError('Pool must be "max" or "avg"')

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4), stride=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8), stride=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8), stride=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8), stride=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8), stride=(8, 8))

        reduction = nn.SequentialCell(
            nn.Conv2d(2048, feats, 1, has_bias=False, weight_init='HeNormal'),
            nn.BatchNorm2d(feats, gamma_init='Normal', beta_init='Zero'),
            nn.ReLU(),
        )

        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        self.fc_id_2048_0 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.fc_id_2048_1 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.fc_id_2048_2 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')

        self.fc_id_256_1_0 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.fc_id_256_1_1 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.fc_id_256_2_0 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.fc_id_256_2_1 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.fc_id_256_2_2 = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')

        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        """ Forward """
        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(axis=3).squeeze(axis=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(axis=3).squeeze(axis=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(axis=3).squeeze(axis=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(axis=3).squeeze(axis=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(axis=3).squeeze(axis=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(axis=3).squeeze(axis=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(axis=3).squeeze(axis=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(axis=3).squeeze(axis=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = self.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3])

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
