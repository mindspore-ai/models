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
"""refinenet"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import set_seed
set_seed(1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=1, has_bias=bias)


class CRPBlock(nn.Cell):
    """chained residual pooling"""
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        layers = []
        for i in range(n_stages):
            if i == 0:
                layer = conv3x3(in_planes, out_planes, stride=1, bias=False)
            else:
                layer = conv3x3(out_planes, out_planes, stride=1, bias=False)
            layers.append(layer)
        self.layers = nn.CellList(layers)
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, pad_mode='same')

    def construct(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = self.layers[i](top)
            x = top + x
        return x


class RCUBlock(nn.Cell):
    """Residual Conv Unit"""
    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        layers = []
        for i in range(n_blocks):
            seq = nn.SequentialCell([])
            for j in range(n_stages):
                if j == 0:
                    relu1 = nn.ReLU()
                    if i == 0:
                        con1 = conv3x3(in_planes, out_planes, stride=1, bias=True)
                    else:
                        con1 = conv3x3(out_planes, out_planes, stride=1, bias=True)
                    seq.append(relu1)
                    seq.append(con1)
                else:
                    relu2 = nn.ReLU()
                    con2 = conv3x3(out_planes, out_planes, stride=1, bias=False)
                    seq.append(relu2)
                    seq.append(con2)
            layers.append(seq)
        self.layers = nn.CellList(layers)
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def construct(self, x):
        for i in range(self.n_blocks):
            residual = x
            x = self.layers[i](x)
            x += residual
        return x


class Bottleneck(nn.Cell):
    """bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_batch_statistics=False, weights_update=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.use_batch_statistics = use_batch_statistics
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=self.use_batch_statistics)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=self.use_batch_statistics)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, use_batch_statistics=self.use_batch_statistics)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        if not weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False
            self.downsample[0].weight.requires_grad = False

    def construct(self, x):
        """construct"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class RefineNet(nn.Cell):
    """network"""
    def __init__(self, block, layers, num_classes=21, use_batch_statistics=False):
        self.inplanes = 64
        super(RefineNet, self).__init__()
        self.do4 = nn.Dropout(p=0.0)
        self.do3 = nn.Dropout(p=0.0)
        self.do = nn.Dropout(p=0.0)
        self.use_batch_statistics = use_batch_statistics
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, use_batch_statistics=self.use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0], weights_update=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, weights_update=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  pad_mode='pad', padding=1, has_bias=True)
        self.resize = nn.ResizeBilinear()

    def _make_crp(self, in_planes, out_planes, stages):
        """make_crp"""
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.SequentialCell(layers)

    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        """make_rcu"""
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.SequentialCell(layers)

    def _make_layer(self, block, planes, blocks, stride=1, weights_update=True):
        """make different layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                      stride=stride, has_bias=False),
                                            nn.BatchNorm2d(planes * block.expansion,
                                                           use_batch_statistics=self.use_batch_statistics)])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, weights_update=weights_update))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        resize_shape = ops.Shape()(x)[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do4(l4)
        l3 = self.do3(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        resize_shape3 = ops.Shape()(l3)[2:]
        x4 = self.resize(x4, resize_shape3, align_corners=True)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = self.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        resize_shape2 = ops.Shape()(l2)[2:]
        x3 = self.resize(x3, resize_shape2, align_corners=True)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = self.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        resize_shape1 = ops.Shape()(l1)[2:]
        x2 = self.resize(x2, size=resize_shape1, align_corners=True)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = self.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)

        logits = self.clf_conv(x1)
        logits = self.resize(logits, resize_shape, align_corners=False)
        return logits
