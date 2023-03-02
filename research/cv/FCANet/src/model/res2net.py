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
"""Res2Net Backbone"""
import math
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, weight_init="he_normal"
    )


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """3x3 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        pad_mode="same",
        padding=0,
        dilation=dilation,
        weight_init="he_normal",
    )


class Resnet(nn.Cell):
    """ official resnet"""

    def __init__(
            self,
            block,
            block_num,
            output_stride=16,
            use_batch_statistics=True,
            input_channels=3,
    ):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1_0 = nn.Conv2d(
            input_channels,
            32,
            3,
            stride=2,
            pad_mode="same",
            padding=0,
            weight_init="he_normal",
        )
        self.bn1_0 = nn.BatchNorm2d(
            32, eps=1e-4, use_batch_statistics=use_batch_statistics
        )

        self.conv1_1 = nn.Conv2d(
            32, 32, 3, stride=1, pad_mode="same", padding=0, weight_init="he_normal"
        )

        self.bn1_1 = nn.BatchNorm2d(
            32, eps=1e-4, use_batch_statistics=use_batch_statistics
        )
        self.conv1_2 = nn.Conv2d(
            32,
            self.inplanes,
            3,
            stride=1,
            pad_mode="same",
            padding=0,
            weight_init="he_normal",
        )

        self.bn1 = nn.BatchNorm2d(
            self.inplanes, eps=1e-4, use_batch_statistics=use_batch_statistics
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(
            block, 64, block_num[0], use_batch_statistics=use_batch_statistics
        )
        self.layer2 = self._make_layer(
            block,
            128,
            block_num[1],
            stride=2,
            use_batch_statistics=use_batch_statistics,
        )

        if output_stride == 16:
            self.layer3 = self._make_layer(
                block,
                256,
                block_num[2],
                stride=2,
                use_batch_statistics=use_batch_statistics,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                block_num[3],
                stride=1,
                base_dilation=2,
                grids=[1, 2, 4],
                use_batch_statistics=use_batch_statistics,
            )
        elif output_stride == 8:
            self.layer3 = self._make_layer(
                block,
                256,
                block_num[2],
                stride=1,
                base_dilation=2,
                use_batch_statistics=use_batch_statistics,
            )
            self.layer4 = self._make_layer(
                block,
                512,
                block_num[3],
                stride=1,
                base_dilation=4,
                grids=[1, 2, 4],
                use_batch_statistics=use_batch_statistics,
            )

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            base_dilation=1,
            grids=None,
            use_batch_statistics=True,
    ):
        """ res2net make_layer"""
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.SequentialCell(
                    [
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        nn.BatchNorm2d(
                            planes * block.expansion,
                            eps=1e-4,
                            use_batch_statistics=use_batch_statistics,
                        ),
                    ]
                )
            else:
                downsample = nn.SequentialCell(
                    [
                        nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
                        conv1x1(self.inplanes, planes * block.expansion, stride=1),
                        nn.BatchNorm2d(
                            planes * block.expansion,
                            eps=1e-4,
                            use_batch_statistics=use_batch_statistics,
                        ),
                    ]
                )

        if grids is None:
            grids = [1] * blocks

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                dilation=base_dilation * grids[0],
                use_batch_statistics=use_batch_statistics,
                stype="stage",
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=base_dilation * grids[i],
                    use_batch_statistics=use_batch_statistics,
                )
            )

        return nn.SequentialCell(layers)

    def construct(self, x):
        """ res2net construct"""
        x = self.conv1_0(x)
        x = self.bn1_0(x)
        x = self.relu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        out = self.conv1_2(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        l1 = self.layer1(out)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1, l2, l3, l4

    def load_pretrained_model(self, ckpt_file):
        """ load res2net pretrained model"""
        ms_ckpt = load_checkpoint(ckpt_file, net=None)
        weights = {}
        for msname in ms_ckpt:
            param_name = msname
            if "layer1" in param_name:
                if "down_sample_layer.0" in param_name:
                    param_name = param_name.replace(
                        "down_sample_layer.0", "downsample.0"
                    )
                if "down_sample_layer.1" in param_name:
                    param_name = param_name.replace(
                        "down_sample_layer.1", "downsample.1"
                    )
            elif "layer4" in param_name:
                if "down_sample_layer.1" in param_name:
                    param_name = param_name.replace(
                        "down_sample_layer.1", "downsample.0"
                    )
                if "down_sample_layer.2" in param_name:
                    param_name = param_name.replace(
                        "down_sample_layer.2", "downsample.1"
                    )
            else:
                if "down_sample_layer.1" in param_name:
                    param_name = param_name.replace(
                        "down_sample_layer.1", "downsample.1"
                    )
                if "down_sample_layer.2" in param_name:
                    param_name = param_name.replace(
                        "down_sample_layer.2", "downsample.2"
                    )
            weights[param_name] = ms_ckpt[msname].data.asnumpy()

        parameter_dict = {}
        for name in weights:
            parameter_dict[name] = Parameter(
                Tensor(weights[name], mstype.float32), name=name
            )

        tmp = self.conv1_0.weight
        tmp[:, :3, :, :] = parameter_dict["conv1_0.weight"]
        parameter_dict["conv1_0.weight"] = tmp

        param_not_load, _ = load_param_into_net(self, parameter_dict)

        print(
            "Load pretrained model from [{}]!([{}] not load!)".format(
                ckpt_file, len(param_not_load)
            )
        )


class Bottle2neck(nn.Cell):
    """ res2net block"""

    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            dilation=1,
            use_batch_statistics=True,
            baseWidth=26,
            scale=4,
            stype="normal",
    ):
        super(Bottle2neck, self).__init__()
        assert scale > 1, "Res2Net is ResNet when scale = 1"
        width = int(
            math.floor(planes * self.expansion // self.expansion * (baseWidth / 64.0))
        )
        channel = width * scale
        self.conv1 = conv1x1(inplanes, channel)
        self.bn1 = nn.BatchNorm2d(
            channel, eps=1e-4, use_batch_statistics=use_batch_statistics
        )

        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, pad_mode="same")

        self.convs = nn.CellList()
        self.bns = nn.CellList()
        for _ in range(scale - 1):
            self.convs.append(conv3x3(width, width, stride, dilation, dilation))
            self.bns.append(
                nn.BatchNorm2d(
                    width, eps=1e-4, use_batch_statistics=use_batch_statistics
                )
            )

        self.conv3 = conv1x1(channel, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, eps=1e-4, use_batch_statistics=use_batch_statistics
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

        self.add = P.Add()
        self.scale = scale
        self.width = width
        self.stride = stride
        self.stype = stype
        self.split = P.Split(axis=1, output_num=scale)
        self.cat = P.Concat(axis=1)

    def construct(self, x):
        """ bottle2neck construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = self.split(out)

        sp = self.convs[0](spx[0])
        sp = self.relu(self.bns[0](sp))
        out = sp

        for i in range(1, self.scale - 1):
            if self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp[:, :, :, :]
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = self.cat((out, sp))

        if self.stype == "normal":
            out = self.cat((out, spx[self.scale - 1]))
        elif self.stype == "stage":
            out = self.cat((out, self.pool(spx[self.scale - 1])))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


def res2net101(output_stride=16, use_batch_statistics=None, input_channels=3):
    """ res2net101 """
    return Resnet(
        Bottle2neck,
        [3, 4, 23, 3],
        output_stride=output_stride,
        use_batch_statistics=use_batch_statistics,
        input_channels=input_channels,
    )
