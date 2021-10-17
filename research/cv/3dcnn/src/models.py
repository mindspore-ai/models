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
"""
python models.py
"""
import mindspore.nn as nn
import mindspore.ops as ops

from .initializer import GlorotNormal


class DenseNetUnit3DLayer(nn.Cell):
    """
    Define DenseNetUnit3DLayer for Dense24.
    """

    def __init__(self, num_input_features, growth_rate, ksize, bn_decay=0.99):
        super(DenseNetUnit3DLayer, self).__init__()
        self.bn3d = nn.BatchNorm3d(num_features=num_input_features,
                                   eps=0.001,
                                   momentum=bn_decay)
        self.relu = nn.ReLU()
        self.conv3d = nn.Conv3d(num_input_features,
                                out_channels=growth_rate,
                                kernel_size=ksize,
                                pad_mode='same',
                                weight_init=GlorotNormal(),
                                has_bias=False)

    def construct(self, x):
        """
        Construct  DenseNetUnit3DLayer
        """
        x = self.bn3d(x)
        x = self.relu(x)
        x = self.conv3d(x)

        return x


class DenseNetUnit3D(nn.Cell):
    """
    Define DenseNetUnit3D for Dense24.
    """

    def __init__(self, num_input_features, growth_rate, ksize, rep, bn_decay=0.99):
        super(DenseNetUnit3D, self).__init__()
        self.cell_list = nn.CellList()
        for i in range(rep):
            layer = DenseNetUnit3DLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                ksize=ksize,
                bn_decay=bn_decay,
            )
            self.cell_list.append(layer)
            self.concatenate = ops.Concat(axis=1)

    def construct(self, x):
        """
        Construct  DenseNetUnit3D
        """
        for layer in self.cell_list:
            concat = x
            x = layer(x)
            x = self.concatenate((concat, x))

        return x


class Conv3DWithBN(nn.Cell):
    """
    Define Conv3DWithBN for Dense24.
    """

    def __init__(self, in_channels, out_channels, ksize, strides, padding='same', dilation_rate=1, decay=0.99):
        super(Conv3DWithBN, self).__init__()
        self.con3d = nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=ksize,
                               stride=strides,
                               pad_mode=padding,
                               dilation=dilation_rate,
                               has_bias=False,
                               weight_init='HeNormal')
        self.bn3d = nn.BatchNorm3d(num_features=out_channels,
                                   eps=0.001,
                                   momentum=decay)
        self.relu = nn.ReLU()

    def construct(self, x):
        """
        Construct  Conv3DWithBN
        """
        x = self.con3d(x)
        x = self.bn3d(x)
        x = self.relu(x)

        return x


class BraTS2ScaleDenseNetConcat(nn.Cell):
    """
    Define BraTS2ScaleDenseNetConcat for Dense24.
    """

    def __init__(self):
        super(BraTS2ScaleDenseNetConcat, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2,
                                out_channels=24,
                                kernel_size=3,
                                stride=1,
                                pad_mode='same',
                                has_bias=True,
                                weight_init='XavierUniform')
        self.denseblock1 = DenseNetUnit3D(num_input_features=24,
                                          growth_rate=12,
                                          ksize=3,
                                          rep=6)
        self.bn3d_1 = nn.BatchNorm3d(num_features=96,
                                     eps=0.001,
                                     momentum=0.99)
        self.out_15_postconv = Conv3DWithBN(in_channels=96,
                                            out_channels=96,
                                            ksize=1,
                                            strides=1)
        self.denseblock2 = DenseNetUnit3D(num_input_features=96,
                                          growth_rate=12,
                                          ksize=3,
                                          rep=6)
        self.bn3d_2 = nn.BatchNorm3d(num_features=168,
                                     eps=0.001,
                                     momentum=0.99)
        self.relu = nn.ReLU()
        self.out_27_postconv = Conv3DWithBN(in_channels=168,
                                            out_channels=168,
                                            ksize=1,
                                            strides=1)

    def construct(self, x):
        """
        Construct  BraTS2ScaleDenseNetConcat
        """
        x = self.conv3d(x)
        x = self.denseblock1(x)

        out_15rf = self.bn3d_1(x)
        out_15rf = self.relu(out_15rf)
        out_15rf = self.out_15_postconv(out_15rf)

        x = self.denseblock2(x)
        out_27rf = self.bn3d_2(x)
        out_27rf = self.relu(out_27rf)
        out_27rf = self.out_27_postconv(out_27rf)

        return out_15rf, out_27rf


class Dense24(nn.Cell):
    """
    Dense24 architecture.
    """

    def __init__(self, num_classes=5):
        super(Dense24, self).__init__()
        self.flair = BraTS2ScaleDenseNetConcat()
        self.t1 = BraTS2ScaleDenseNetConcat()
        self.concatenate = ops.Concat(axis=1)
        self.flair_t2_15_cls = nn.Conv3d(in_channels=96,
                                         out_channels=2,
                                         kernel_size=1,
                                         stride=1,
                                         pad_mode='same',
                                         has_bias=True,
                                         weight_init='XavierUniform')
        self.flair_t2_27_cls = nn.Conv3d(in_channels=168,
                                         out_channels=2,
                                         kernel_size=1,
                                         stride=1,
                                         pad_mode='same',
                                         has_bias=True,
                                         weight_init='XavierUniform')
        self.t1_t1ce_15_cls = nn.Conv3d(in_channels=192,
                                        out_channels=num_classes,
                                        kernel_size=1,
                                        stride=1,
                                        pad_mode='same',
                                        has_bias=True,
                                        weight_init='XavierUniform')
        self.t1_t1ce_27_cls = nn.Conv3d(in_channels=336,
                                        out_channels=num_classes,
                                        kernel_size=1,
                                        stride=1,
                                        pad_mode='same',
                                        has_bias=True,
                                        weight_init='XavierUniform')

    def construct(self, flair_t2_node, t1_t1ce_node):
        """
        Construct  Dense24
        """
        flair_t2_15, flair_t2_27 = self.flair(flair_t2_node)
        t1_t1ce_15, t1_t1ce_27 = self.t1(t1_t1ce_node)
        t1_t1ce_15 = self.concatenate((t1_t1ce_15, flair_t2_15))
        t1_t1ce_27 = self.concatenate((t1_t1ce_27, flair_t2_27))
        flair_t2_15 = self.flair_t2_15_cls(flair_t2_15)
        flair_t2_27 = self.flair_t2_27_cls(flair_t2_27)
        t1_t1ce_15 = self.t1_t1ce_15_cls(t1_t1ce_15)
        t1_t1ce_27 = self.t1_t1ce_27_cls(t1_t1ce_27)
        flair_t2_score = flair_t2_15[:, :, 13:25, 13:25, 13:25] + \
                         flair_t2_27[:, :, 13:25, 13:25, 13:25]
        t1_t1ce_score = t1_t1ce_15[:, :, 13:25, 13:25, 13:25] + \
                        t1_t1ce_27[:, :, 13:25, 13:25, 13:25]

        return flair_t2_score, t1_t1ce_score
