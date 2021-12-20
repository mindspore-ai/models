"""
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
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net

from src.resnet50 import ResNet50



def Upsample_resize(source, target, device_target):
    target_shape = target.shape

    resize = None
    if device_target == "GPU":
        resize = ops.ResizeNearestNeighbor((target_shape[-1], target_shape[-2]))
    else:
        resize = ops.ResizeBilinear((target_shape[-1], target_shape[-2]))

    return resize(source)



class MSCM(nn.Cell):
    """
    MSCM processing module in the original paper

    """
    def __init__(self, in_c, out_c):
        super(MSCM, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1), has_bias=True)
        self.branch1 = nn.SequentialCell(nn.Conv2d(in_channels=out_c, out_channels=out_c,
                                                   kernel_size=(1, 1), has_bias=True),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                                                   pad_mode='pad', padding=1, dilation=1, has_bias=True),
                                         nn.ReLU())

        self.branch2 = nn.SequentialCell(nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                                                   pad_mode='pad', padding=1, dilation=1, has_bias=True),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                                                   pad_mode='pad', padding=2, dilation=2, has_bias=True),
                                         nn.ReLU())

        self.branch3 = nn.SequentialCell(nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(5, 5),
                                                   pad_mode='pad', padding=2, dilation=1, has_bias=True),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                                                   pad_mode='pad', padding=4, dilation=4, has_bias=True),
                                         nn.ReLU())

        self.branch4 = nn.SequentialCell(nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(7, 7),
                                                   pad_mode='pad', padding=3, dilation=1, has_bias=True),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                                                   pad_mode='pad', padding=6, dilation=6, has_bias=True),
                                         nn.ReLU())

        self.conv_end = nn.Conv2d(in_channels=out_c*4, out_channels=1, kernel_size=(3, 3),
                                  pad_mode='pad', padding=1, has_bias=True)

    def construct(self, data):
        x = self.conv(data)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x_ = ops.Concat(1)((x1, x2, x3, x4))
        output = self.conv_end(x_)
        return output


class RA(nn.Cell):
    """
    RA processing module in the original paper
    """
    def __init__(self, in_c, out_c, device_target):
        super(RA, self).__init__()
        self.device_target = device_target
        self.out_c = out_c
        self.convert = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1), has_bias=True)
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                      pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                      pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3),
                      pad_mode='pad', padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=(3, 3), pad_mode='pad', padding=1, has_bias=True)
            )

    def construct(self, x, y):
        y = Upsample_resize(y, x, self.device_target)
        a = ops.Sigmoid()(-y)
        out = ops.BroadcastTo((-1, self.out_c, a.shape[-1], a.shape[-2]))(a)
        x = self.convert(x)
        result = ops.Mul()(x, out)
        result = self.conv(result)
        output = ops.AddN()((result, y))
        return output

class BoneModel(nn.Cell):
    """
    Whole construct of The RAS
    """
    def __init__(self, device_target, pretrained_model):
        super(BoneModel, self).__init__()
        in_c = 2048
        self.device_target = device_target
        self.pre_trained_model_path = pretrained_model
        self.resnet = ResNet50()
        self.mscm = MSCM(in_c=in_c, out_c=64)
        self.ra1 = RA(in_c=64, out_c=64, device_target=device_target)
        self.ra2 = RA(in_c=256, out_c=64, device_target=device_target)
        self.ra3 = RA(in_c=512, out_c=64, device_target=device_target)
        self.ra4 = RA(in_c=1024, out_c=64, device_target=device_target)

        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(ms.Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype(np.float32)))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(ms.Tensor(np.ones(m.gamma.data.shape, dtype=np.float32)))
                m.beta.set_data(ms.Tensor(np.zeros(m.beta.data.shape, dtype=np.float32)))

        self.weight_init_resnet()


    def weight_init_resnet(self):
        """

        initial parameters for resnet50

        """
        param_dict = load_checkpoint(self.pre_trained_model_path)
        pre_param = {}
        param_list = []
        for item in self.resnet.get_parameters():
            param_list.append(item.name)
        for key in param_dict.keys():
            key_new = 'resnet.' + key
            if key_new in param_list:
                pre_param[key_new] = param_dict[key]

        load_param_into_net(self.resnet, pre_param)

    def construct(self, data):
        """

        Args:
            data: Tensor

        Returns:
            5 outputs

        """
        x1, x2, x3, x4, x5 = self.resnet(data)
        mscm_output = self.mscm(x5)
        ra4_out = self.ra4(x4, mscm_output)
        ra3_out = self.ra3(x3, ra4_out)
        ra2_out = self.ra2(x2, ra3_out)
        ra1_out = self.ra1(x1, ra2_out)

        out5 = Upsample_resize(mscm_output, data, self.device_target)
        out4 = Upsample_resize(ra4_out, data, self.device_target)
        out3 = Upsample_resize(ra3_out, data, self.device_target)
        out2 = Upsample_resize(ra2_out, data, self.device_target)
        out1 = Upsample_resize(ra1_out, data, self.device_target)

        return out1, out2, out3, out4, out5
