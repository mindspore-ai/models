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
""" PSPNet """
from src.model.resnet import resnet50
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_param_into_net, load_checkpoint
import mindspore.common.initializer as weight_init


class ResNet(nn.Cell):
    """ The pretrained ResNet """

    def __init__(self, pretrained_path, pretrained=False, deep_base=False, BatchNorm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        resnet = resnet50(deep_base=deep_base, BatchNorm_layer=BatchNorm_layer)
        if pretrained:
            params = load_checkpoint(pretrained_path)
            load_param_into_net(resnet, params)
        if deep_base:
            self.layer1 = nn.SequentialCell(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2,
                                            resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        else:
            self.layer1 = nn.SequentialCell(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4

    def construct(self, x):
        """ ResNet process """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_aux = self.layer4(x)
        x = self.layer5(x_aux)

        return x_aux, x


class AdaPool1(nn.Cell):
    """ 1x1 pooling """

    def __init__(self):
        super(AdaPool1, self).__init__()
        self.reduceMean = ops.ReduceMean(keep_dims=True)

    def construct(self, X):
        """ 1x1 pooling process """
        pooled_1x1 = self.reduceMean(X, (-2, -1))
        return pooled_1x1


class AdaPool2(nn.Cell):
    """ 2x2 pooling """

    def __init__(self):
        super(AdaPool2, self).__init__()
        self.reduceMean = ops.ReduceMean()
        self.reshape = ops.Reshape()

    def construct(self, X):
        """ 2x2 pooling process """
        batch_size, channels, _, _ = X.shape
        X = self.reshape(X, (batch_size, channels, 2, 30, 2, 30))
        pooled_2x2_out = self.reduceMean(X, (3, 5))
        return pooled_2x2_out


class AdaPool3(nn.Cell):
    """ 3x3 pooling """

    def __init__(self):
        super(AdaPool3, self).__init__()
        self.reduceMean = ops.ReduceMean()
        self.reshape = ops.Reshape()

    def construct(self, X):
        """ 3x3 pooling process """
        batch_size, channels, _, _ = X.shape
        X = self.reshape(X, (batch_size, channels, 3, 20, 3, 20))
        pooled_3x3_out = self.reduceMean(X, (3, 5))
        return pooled_3x3_out


class _PSPModule(nn.Cell):
    """ PSP module """

    def __init__(self, in_channels, pool_sizes, feature_shape, BatchNorm_layer=nn.BatchNorm2d):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.BatchNorm_layer = BatchNorm_layer
        self.stage1 = nn.SequentialCell(
            AdaPool1(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False),
            self.BatchNorm_layer(out_channels),
            nn.ReLU(),
        )
        self.stage2 = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False),
            self.BatchNorm_layer(out_channels),
            nn.ReLU()
        )
        self.stage3 = nn.SequentialCell(
            AdaPool3(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False),
            self.BatchNorm_layer(out_channels),
            nn.ReLU(),
        )
        self.stage4 = nn.SequentialCell(
            nn.AvgPool2d(kernel_size=10, stride=10),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False),
            self.BatchNorm_layer(out_channels),
            nn.ReLU()
        )
        self.cat = ops.Concat(axis=1)
        self.feature_shape = feature_shape
        self.resize_ops = ops.ResizeBilinear(
            (self.feature_shape[0], self.feature_shape[1]), True
        )
        self.cast = ops.Cast()

    def construct(self, x):
        """ PSP module process """
        x = self.cast(x, mindspore.float32)
        s1_out = self.resize_ops(self.stage1(x))
        s2_out = self.resize_ops(self.stage2(x))
        s3_out = self.resize_ops(self.stage3(x))
        s4_out = self.resize_ops(self.stage4(x))
        out = (x, s1_out, s2_out, s3_out, s4_out)
        out = self.cat(out)

        return out


class PSPNet(nn.Cell):
    """ PSPNet """

    def __init__(
            self,
            pool_sizes=None,
            feature_size=60,
            num_classes=21,
            backbone="resnet50",
            pretrained=True,
            pretrained_path="",
            aux_branch=False,
            deep_base=False,
            BatchNorm_layer=nn.BatchNorm2d
    ):
        """
        """
        super(PSPNet, self).__init__()
        if pool_sizes is None:
            pool_sizes = [1, 2, 3, 6]
        if backbone == "resnet50":
            self.backbone = ResNet(
                pretrained=pretrained,
                pretrained_path=pretrained_path,
                deep_base=deep_base,
                BatchNorm_layer=BatchNorm_layer
            )
            aux_channel = 1024
            out_channel = 2048
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use resnet50 .".format(backbone)
            )
        self.BatchNorm_layer = BatchNorm_layer
        self.feature_shape = [feature_size, feature_size]
        self.pool_sizes = [feature_size // pool_size for pool_size in pool_sizes]
        self.ppm = _PSPModule(in_channels=out_channel, pool_sizes=self.pool_sizes, feature_shape=self.feature_shape)
        self.cls = nn.SequentialCell(
            nn.Conv2d(out_channel * 2, 512, kernel_size=3, padding=1, pad_mode="pad", has_bias=False),
            self.BatchNorm_layer(512),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Conv2d(512, num_classes, kernel_size=1, has_bias=True)
        )
        self.aux_branch = aux_branch
        if self.aux_branch:
            self.auxiliary_branch = nn.SequentialCell(
                nn.Conv2d(aux_channel, 256, kernel_size=3, padding=1, pad_mode="pad", has_bias=False),
                self.BatchNorm_layer(256),
                nn.ReLU(),
                nn.Dropout(0.9),
                nn.Conv2d(256, num_classes, kernel_size=1, has_bias=True)
            )
        self.resize = nn.ResizeBilinear()
        self.shape = ops.Shape()
        self.init_weights(self.cls)

    def init_weights(self, *models):
        """ init the model parameters """
        for model in models:
            for _, cell in model.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight.set_data(
                        weight_init.initializer(
                            weight_init.HeNormal(), cell.weight.shape, cell.weight.dtype
                        )
                    )
                if isinstance(cell, nn.Dense):
                    cell.weight.set_data(
                        weight_init.initializer(
                            weight_init.TruncatedNormal(0.01),
                            cell.weight.shape,
                            cell.weight.dtype,
                        )
                    )
                    cell.bias.set_data(1e-4, cell.bias.shape, cell.bias.dtype)

    def construct(self, x):
        """ PSPNet process """
        x_shape = self.shape(x)
        x_aux, x = self.backbone(x)
        x = self.ppm(x)
        out = self.cls(x)
        out = self.resize(out, size=(x_shape[2:4]), align_corners=True)
        if self.aux_branch:
            out_aux = self.auxiliary_branch(x_aux)
            output_aux = self.resize(out_aux, size=(x_shape[2:4]), align_corners=True)
            return output_aux, out
        return out
