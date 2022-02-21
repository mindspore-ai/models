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
'''model of UNet3+'''
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor

class unetConv2(nn.Cell):
    '''unetConv2'''
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1, weight_init="HeNormal"):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        conv_layer = []
        if is_batchnorm:
            for _ in range(1, n + 1):
                conv_layer.extend([
                    nn.Conv2d(in_size, out_size, ks, s, pad_mode="pad", padding=p, weight_init="HeNormal"),
                    nn.BatchNorm2d(out_size, gamma_init="ones"),
                    nn.ReLU()
                ])
                in_size = out_size
        else:
            for _ in range(1, n + 1):
                conv_layer.extend([
                    nn.Conv2d(in_size, out_size, ks, s, pad_mode="pad", padding=p, weight_init="HeNormal"),
                    nn.ReLU()
                ])
                in_size = out_size
        self.conv = nn.SequentialCell(conv_layer)

    def construct(self, inputs):
        '''construct'''
        return self.conv(inputs)


class UNet3Plus(nn.Cell):
    '''UNet3Plus'''
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4,
                 is_deconv=True, is_batchnorm=True):
        super(UNet3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [16, 32, 64, 128, 256]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, \
                                 pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, \
                                 pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, \
                                 pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.ResizeBilinear = nn.ResizeBilinear()
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, \
                           pad_mode="pad", padding=1, weight_init="HeNormal")  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels, gamma_init="ones")
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, \
                                 pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, \
                                 pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, \
                           pad_mode="pad", padding=1, weight_init="HeNormal")  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels, gamma_init="ones")
        self.relu3d_1 = nn.ReLU()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, \
                                 pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, \
                           pad_mode="pad", padding=1, weight_init="HeNormal")  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels, gamma_init="ones")
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, \
                                  pad_mode="pad", padding=1, weight_init="HeNormal")
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels, gamma_init="ones")
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, \
                           pad_mode="pad", padding=1, weight_init="HeNormal")  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels, gamma_init="ones")
        self.relu1d_1 = nn.ReLU()

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, \
                           pad_mode="pad", padding=1, weight_init="HeNormal")
        self.concat1 = ops.Concat(1)
    def construct(self, inputs):
        '''construct'''
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(\
                     self.ResizeBilinear(hd5, scale_factor=2, align_corners=True))))


        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            self.concat1((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4))))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(\
                     self.ResizeBilinear(hd4, scale_factor=2, align_corners=True))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(\
                     self.ResizeBilinear(hd5, scale_factor=4, align_corners=True))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            self.concat1((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3))))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(\
                     self.ResizeBilinear(hd3, scale_factor=2, align_corners=True))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(\
                     self.ResizeBilinear(hd4, scale_factor=4, align_corners=True))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(\
                     self.ResizeBilinear(hd5, scale_factor=8, align_corners=True))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            self.concat1((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2))))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(\
                     self.ResizeBilinear(hd2, scale_factor=2, align_corners=True))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(\
                     self.ResizeBilinear(hd3, scale_factor=4, align_corners=True))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(\
                     self.ResizeBilinear(hd4, scale_factor=8, align_corners=True))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(\
                     self.ResizeBilinear(hd5, scale_factor=16, align_corners=True))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            self.concat1((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1))))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return d1

class BCEDiceLoss(nn.Cell):
    '''BCEDiceLoss'''
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bceloss = ops.BinaryCrossEntropy()
        self.sigmoid = ops.Sigmoid()
        self.reduceSum = ops.ReduceSum(keep_dims=False)
        self.one_tensor = Tensor(np.ones([2, 1, 512, 512]), mindspore.float32)
    def construct(self, predict, target):
        '''construct'''
        bce = self.bceloss(self.sigmoid(predict), target, self.one_tensor)
        smooth = 1e-5
        predict = self.sigmoid(predict)
        num = target.shape[0]
        predict = predict.view(num, -1)
        target = target.view(num, -1)
        intersection = (predict * target)
        dice = (2. * self.reduceSum(intersection, 1) + smooth) / \
               (self.reduceSum(predict, 1) + self.reduceSum(target, 1) + smooth)
        dice = 1 - dice / num
        return 0.5 * bce + dice

class UNet3PlusWithLossCell(nn.Cell):
    '''UNet3PlusWithLossCell'''
    def __init__(self, network):
        super(UNet3PlusWithLossCell, self).__init__()
        self.network = network
        self.loss = BCEDiceLoss()
    def construct(self, image, mask):
        '''construct'''
        output = self.network(image)
        return self.loss(output, mask)
