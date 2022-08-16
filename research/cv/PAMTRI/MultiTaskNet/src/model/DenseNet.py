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
"""DenseNet"""
from collections import OrderedDict
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import initializer as init
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    """conv7x7"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")

def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    """conv3x3"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")

def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    """conv1x1"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")

class GlobalAvgPooling(nn.Cell):
    """GlobalAvgPooling"""
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.mean(x, (2, 3))
        b, c, _, _ = self.shape(x)
        x = self.reshape(x, (b, c))
        return x

class CommonHead(nn.Cell):
    """CommonHead"""
    def __init__(self, num_classes, out_channels):
        super(CommonHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = nn.Dense(out_channels, num_classes, has_bias=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)

        return x

class _DenseLayer(nn.Cell):
    """_DenseLayer"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1x1(num_input_features, bn_size*growth_rate)

        self.norm2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(bn_size*growth_rate, growth_rate)

        self.keep_prob = 1.0 - drop_rate
        self.dropout = nn.Dropout(keep_prob=self.keep_prob)

    def construct(self, features):
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))

        if self.keep_prob < 1:
            new_features = self.dropout(new_features)

        return new_features

class _DenseBlock(nn.Cell):
    """_DenseBlock"""
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.cell_list = nn.CellList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.cell_list.append(layer)

        self.concate = P.Concat(axis=1)

    def construct(self, init_features):
        features = init_features
        for layer in self.cell_list:
            new_features = layer(features)
            features = self.concate((features, new_features))

        return features

class _Transition(nn.Cell):
    """_Transition"""
    def __init__(self, num_input_features, num_output_features, avgpool=False):
        super(_Transition, self).__init__()
        poollayer = nn.AvgPool2d(kernel_size=2, stride=2)

        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm2d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', conv1x1(num_input_features, num_output_features)),
            ('pool', poollayer)
        ]))

    def construct(self, x):
        x = self.features(x)

        return x

class DenseNet(nn.Cell):
    """DenseNet"""
    __constants__ = ['features']

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        layers = OrderedDict()
        if num_init_features:
            layers['conv0'] = conv7x7(3, num_init_features, stride=2, padding=3)
            layers['norm0'] = nn.BatchNorm2d(num_init_features)
            layers['relu0'] = nn.ReLU()
            layers['pool0'] = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
            num_features = num_init_features
        else:
            layers['conv0'] = conv3x3(3, growth_rate*2, stride=1, padding=1)
            layers['norm0'] = nn.BatchNorm2d(growth_rate*2)
            layers['relu0'] = nn.ReLU()
            num_features = growth_rate * 2

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            layers['denseblock%d'%(i+1)] = block
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                if num_init_features:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2, avgpool=False)
                else:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2, avgpool=True)
                layers['transition%d'%(i+1)] = trans
                num_features = num_features // 2

        layers['norm5'] = nn.BatchNorm2d(num_features)

        self.features = nn.SequentialCell(layers)
        self.out_channels = num_features

        self.relu = nn.ReLU()
        self.avgpool = GlobalAvgPooling()
        self.classifier = nn.Dense(self.out_channels, num_classes, has_bias=True)

    def construct(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out

class DenseNet121(nn.Cell):
    """DenseNet121"""
    def __init__(self, pretrain_path, num_vids, num_vcolors=10, num_vtypes=9,
                 keyptaware=True, heatmapaware=True, segmentaware=True, multitask=True, is_pretrained=True):
        super(DenseNet121, self).__init__()
        self.keyptaware = keyptaware
        self.multitask = multitask

        densenet121 = DenseNet()
        if is_pretrained:
            param_dict = load_checkpoint(pretrain_path)
            load_param_into_net(densenet121, param_dict)

        self.base = densenet121.features

        num_channels = 3
        if heatmapaware:
            num_channels += 36
        if segmentaware:
            num_channels += 13

        if num_channels > 3:
            if is_pretrained:
                pretrained_weights = densenet121.features[0].weight
                self.base[0] = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2,
                                         has_bias=False, padding=3, pad_mode="pad")
                weight_init = init.initializer(init.Normal(sigma=0.02), \
                    self.base[0].weight.shape, self.base[0].weight.dtype)
                weight_init[:, :3, :, :] = pretrained_weights
                self.base[0].weight.set_data(weight_init)

            else:
                self.base[0] = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2,
                                         has_bias=False, padding=3, pad_mode="pad")

        self.leaky_relu = nn.LeakyReLU(alpha=0.01)
        self.concate = P.Concat(axis=1)
        self.ave_pool = nn.AvgPool2d(kernel_size=(8, 8), stride=(8, 8))

        if self.keyptaware and self.multitask:
            self.fc_vid = nn.Dense(1024 + 108, 1024, has_bias=True)
            self.fc_vcolor = nn.Dense(1024 + 108, 512, has_bias=True)
            self.fc_vtype = nn.Dense(1024 + 108, 512, has_bias=True)
            self.classifier_vid = nn.Dense(1024, num_vids, has_bias=True)
            self.classifier_vcolor = nn.Dense(512, num_vcolors, has_bias=True)
            self.classifier_vtype = nn.Dense(512, num_vtypes, has_bias=True)
        elif self.keyptaware:
            self.fc = nn.Dense(1024 + 108, 1024, has_bias=True)
            self.classifier_vid = nn.Dense(1024, num_vids, has_bias=True)
        elif self.multitask:
            self.fc_vid = nn.Dense(1024, 1024, has_bias=True)
            self.fc_vcolor = nn.Dense(1024, 512, has_bias=True)
            self.fc_vtype = nn.Dense(1024, 512, has_bias=True)
            self.classifier_vid = nn.Dense(1024, num_vids, has_bias=True)
            self.classifier_vcolor = nn.Dense(512, num_vcolors, has_bias=True)
            self.classifier_vtype = nn.Dense(512, num_vtypes, has_bias=True)
        else:
            self.classifier_vid = nn.Dense(1024, num_vids, has_bias=True)

        self.feat_dim = 1024

    def construct(self, x, p=None):
        """model construct"""
        out = self.base(x)
        out = self.ave_pool(out)
        f = out.view(out.shape[0], -1)

        if self.keyptaware and self.multitask:
            f = self.concate((f, p))
            f_vid = self.leaky_relu(self.fc_vid(f))
            f_vcolor = self.leaky_relu(self.fc_vcolor(f))
            f_vtype = self.leaky_relu(self.fc_vtype(f))
            y_id = self.classifier_vid(f_vid)
            y_color = self.classifier_vcolor(f_vcolor)
            y_type = self.classifier_vtype(f_vtype)

            return y_id, y_color, y_type, f_vid
        if self.keyptaware:
            f = self.concate((f, p))
            f = self.leaky_relu(self.fc(f))
            y_id = self.classifier_vid(f)

            return y_id, f
        if self.multitask:
            f_vid = self.leaky_relu(self.fc_vid(f))
            f_vcolor = self.leaky_relu(self.fc_vcolor(f))
            f_vtype = self.leaky_relu(self.fc_vtype(f))
            y_id = self.classifier_vid(f_vid)
            y_color = self.classifier_vcolor(f_vcolor)
            y_type = self.classifier_vtype(f_vtype)

            return y_id, y_color, y_type, f_vid

        y_id = self.classifier_vid(f)

        return y_id, f
