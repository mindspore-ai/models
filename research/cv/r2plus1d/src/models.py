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
"""R(2+1)D Network"""
import mindspore.nn as nn
import mindspore.ops as ops

class Conv2Plus1D(nn.Cell):
    """
    Conv2Plus1D
    args:
        inplanes: Integer
        planes: Integer
        midplanes: Integer
        stride: Integer
        padding: Integer
    """
    def __init__(self,
                 inplanes,
                 planes,
                 midplanes,
                 stride=1,
                 padding=1,
                 norm_kwargs=None,
                 **kwargs):
        super(Conv2Plus1D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inplanes,
                               out_channels=midplanes,
                               kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               pad_mode="pad",
                               padding=(0, 0, padding, padding, padding, padding),
                               has_bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=midplanes,
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=midplanes,
                               out_channels=planes,
                               kernel_size=(3, 1, 1),
                               stride=(stride, 1, 1),
                               pad_mode="pad",
                               padding=(padding, padding, 0, 0, 0, 0),
                               has_bias=False)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class BasicBlock(nn.Cell):
    """
    BasicBlock
    args:
        inplanes: Integer
        planes: Integer
        stride: Integer
        downsample: SequentialCell
        layer_name: String
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_kwargs=None, layer_name='',
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = Conv2Plus1D(inplanes, planes, midplanes, stride)
        self.bn1 = nn.BatchNorm3d(num_features=planes,
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU()
        self.conv2 = Conv2Plus1D(planes, planes, midplanes)
        self.bn2 = nn.BatchNorm3d(num_features=planes,
                                  **({} if norm_kwargs is None else norm_kwargs))

    def construct(self, x):
        """
        construct
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

class R2Plus1D(nn.Cell):
    r"""The R2+1D network.
    A Closer Look at Spatiotemporal Convolutions for Action Recognition.
    CVPR, 2018. https://arxiv.org/abs/1711.11248
    """
    def __init__(self, num_classes, block, layers, dropout_ratio=0.5,
                 num_segment=1, num_crop=1, feat_ext=False,
                 init_std=0.001, partial_bn=False,
                 norm_kwargs=None, **kwargs):
        super(R2Plus1D, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * block.expansion

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=45, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), pad_mode="pad",
                               padding=(0, 0, 3, 3, 3, 3), has_bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=45, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=45,
                               out_channels=64,
                               kernel_size=(3, 1, 1),
                               stride=(1, 1, 1),
                               pad_mode="pad",
                               padding=(1, 1, 0, 0, 0, 0),
                               dilation=1,
                               has_bias=False)

        self.bn2 = nn.BatchNorm3d(num_features=64, **({} if norm_kwargs is None else norm_kwargs))

        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True

        self.layer1 = self._make_res_layer(block=block,
                                           planes=64,
                                           blocks=layers[0],
                                           layer_name='layer1_')
        self.layer2 = self._make_res_layer(block=block,
                                           planes=128,
                                           blocks=layers[1],
                                           stride=2,
                                           layer_name='layer2_')
        self.layer3 = self._make_res_layer(block=block,
                                           planes=256,
                                           blocks=layers[2],
                                           stride=2,
                                           layer_name='layer3_')
        self.layer4 = self._make_res_layer(block=block,
                                           planes=512,
                                           blocks=layers[3],
                                           stride=2,
                                           layer_name='layer4_')

        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.dropout = nn.Dropout(1-self.dropout_ratio)
        self.fc = nn.Dense(in_channels=self.feat_dim, out_channels=num_classes)

    def _make_res_layer(self,
                        block,
                        planes,
                        blocks,
                        stride=1,
                        norm_kwargs=None,
                        layer_name=''):
        """Build each stage of a ResNet"""
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv3d(in_channels=self.inplanes,
                          out_channels=planes * block.expansion,
                          kernel_size=1,
                          stride=(stride, stride, stride),
                          has_bias=False),
                nn.BatchNorm3d(num_features=planes * block.expansion,
                               **({} if norm_kwargs is None else norm_kwargs))])

        layers = []
        layers.append(block(inplanes=self.inplanes,
                            planes=planes,
                            stride=stride,
                            downsample=downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        construct
        """
        bs, _, _, _, _ = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x, (2, 3, 4))
        x = x.view(bs, -1)

        if self.feat_ext:
            return x

        x = self.fc(self.dropout(x))
        return x

def get_r2plus1d_model(num_classes, layer_num=18):
    model_layers = [2, 2, 2, 2] if layer_num == 18 else [3, 4, 6, 3]
    model = R2Plus1D(num_classes=num_classes,
                     block=BasicBlock,
                     layers=model_layers,
                     num_segment=1,
                     num_crop=1,
                     feat_ext=False,
                     partial_bn=False)
    return model
