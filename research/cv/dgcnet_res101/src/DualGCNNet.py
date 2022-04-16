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
"""DGCNet(res101) network."""
import mindspore
import mindspore.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad',
                     padding=1, has_bias=False)


class Bottleneck(nn.Cell):
    """Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad',
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

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

        out = out + residual
        out = self.relu_inplace(out)

        return out


class SpatialGCN(nn.Cell):
    """SpatialGCN"""
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1, pad_mode='valid', has_bias=True)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1, pad_mode='valid', has_bias=True)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1, pad_mode='valid', has_bias=True)
        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, has_bias=False)
        self.bn_wg = nn.BatchNorm2d(inter_plane, use_batch_statistics=True)
        self.softmax = mindspore.ops.Softmax(axis=2)
        self.out = nn.SequentialCell([nn.Conv2d(inter_plane, plane, kernel_size=1, has_bias=True),
                                      nn.BatchNorm2d(plane, use_batch_statistics=True)])
        self.reshape = mindspore.ops.Reshape()
        self.transpose = mindspore.ops.Transpose()
        self.bmm = mindspore.ops.BatchMatMul()
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b, c, h, _ = node_k.shape
        node_k = self.reshape(node_k, (b, c, -1))
        node_k = self.transpose(node_k, (0, 2, 1))
        node_q = self.reshape(node_q, (b, c, -1))
        node_v = self.reshape(node_v, (b, c, -1))
        node_v = self.transpose(node_v, (0, 2, 1))

        AV = self.bmm(node_q, node_v)
        AV = self.softmax(AV)
        AV = self.bmm(node_k, AV)
        AV = self.transpose(AV, (0, 2, 1))
        AVW = self.conv_wg(AV)
        AVW = mindspore.ops.Squeeze(-1)(self.bn_wg(mindspore.ops.ExpandDims()(AVW, -1)))
        AVW = self.reshape(AVW, (b, c, h, -1))
        out = self.relu(self.out(AVW) + x)
        return out


class DualGCN(nn.Cell):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, has_bias=False)
        self.bn_phi = nn.BatchNorm2d(planes // ratio * 2, use_batch_statistics=True)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, has_bias=False)
        self.bn_theta = nn.BatchNorm2d(planes // ratio, use_batch_statistics=True)

        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, has_bias=False)
        self.bn_adj = nn.BatchNorm2d(planes // ratio, use_batch_statistics=True)

        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, has_bias=False)
        self.bn_wg = nn.BatchNorm2d(planes // ratio * 2, use_batch_statistics=True)

        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes, use_batch_statistics=True)

        self.local = nn.SequentialCell([
            nn.Conv2d(planes, planes, 3, group=planes, stride=2, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(planes, use_batch_statistics=True),
            nn.Conv2d(planes, planes, 3, group=planes, stride=2, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(planes, use_batch_statistics=True),
            nn.Conv2d(planes, planes, 3, group=planes, stride=2, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(planes, use_batch_statistics=True)])
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.SequentialCell([nn.Conv2d(planes * 2, planes, kernel_size=1, has_bias=False),
                                        nn.BatchNorm2d(planes, use_batch_statistics=True)])
        self.reshape = mindspore.ops.Reshape()
        self.transpose = mindspore.ops.Transpose()
        self.matmul = nn.MatMul()
        self.relu = nn.ReLU()

    def to_matrix(self, x):
        """to_matrix"""
        n, c, _, _ = x.shape
        shape = (n, c, -1)
        x = self.reshape(x, shape)
        return x

    def construct(self, feat):
        """construct"""
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        resize_bilinear = mindspore.ops.ResizeBilinear(size=x.shape[2:], align_corners=True)
        local = resize_bilinear(local)
        spatial_local_feat = x * local + x

        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)
        btr = self.transpose(b, (0, 2, 1))

        z_idt = self.matmul(x_sqz, btr)

        z = self.transpose(z_idt, (0, 2, 1))

        z = self.conv_adj(z)
        z = mindspore.ops.Squeeze(-1)(self.bn_adj(mindspore.ops.ExpandDims()(z, -1)))
        z = self.transpose(z, (0, 2, 1))

        z += z_idt

        z = self.conv_wg(z)
        z = mindspore.ops.Squeeze(-1)(self.bn_wg(mindspore.ops.ExpandDims()(z, -1)))

        y = self.matmul(z, b)

        n, _, h, w = x.shape
        y = self.reshape(y, (n, -1, h, w))

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = self.relu(x + y)

        cat = mindspore.ops.Concat(1)
        out = self.final(cat((spatial_local_feat, g_out)))

        return out


class DualGCNHead(nn.Cell):
    """DualGCNHead"""
    def __init__(self, inplanes, interplanes, num_classes):
        super(DualGCNHead, self).__init__()
        self.conva = nn.SequentialCell([nn.Conv2d(inplanes, interplanes, 3, pad_mode='pad', padding=1, has_bias=False),
                                        nn.BatchNorm2d(interplanes, use_batch_statistics=True),
                                        nn.ReLU()])
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.SequentialCell([nn.Conv2d(interplanes, interplanes, 3, pad_mode='pad', padding=1,
                                                  has_bias=False),
                                        nn.BatchNorm2d(interplanes, use_batch_statistics=True),
                                        nn.ReLU()])

        self.bottleneck = nn.SequentialCell([
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, pad_mode='pad', padding=1, dilation=1,
                      has_bias=False),
            nn.BatchNorm2d(interplanes, use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
        ])

    def construct(self, x):
        """construct"""
        output = self.conva(x)
        output = self.dualgcn(output)
        output = self.convb(output)
        cat = mindspore.ops.Concat(1)
        output = self.bottleneck(cat([x, output]))
        return output


class ResNet(nn.Cell):
    """ResNet"""
    def __init__(self, block, layers, num_classes, is_train):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.is_train = is_train
        self.conv1 = nn.SequentialCell([
            conv3x3(3, 64, stride=2),
            nn.BatchNorm2d(64, use_batch_statistics=True),
            nn.ReLU(),
            conv3x3(64, 64),
            nn.BatchNorm2d(64, use_batch_statistics=True),
            nn.ReLU(),
            conv3x3(64, 128)])
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 2, 4))
        self.head = DualGCNHead(2048, 512, num_classes)
        self.dsn = nn.SequentialCell([
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(512, use_batch_statistics=True),
            nn.Dropout(0.9),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
        ])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        """make_layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=True)])

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)

        if self.is_train:
            output = [x, x_dsn]
        else:
            output = [x]

        return output


def DualSeg_res101(num_classes=21, is_train=True):
    """DualSeg_res101"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, is_train)

    return model
