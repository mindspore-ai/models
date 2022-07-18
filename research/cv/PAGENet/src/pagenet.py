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
import os
import mindspore.ops as P
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from src.vgg import VGG

class upSampleLike(nn.Cell):

    def __init__(self):
        super(upSampleLike, self).__init__()
        self.resize = nn.ResizeBilinear()
    def construct(self, fm, x1):
        fm = self.resize(fm, (x1.shape[2], x1.shape[3]))
        return fm

class MyMaxPool(nn.Cell):
    def __init__(self, stride):
        super().__init__()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.stride = stride

    def construct(self, x):
        n, c, h, w = x.shape
        x_t = P.reshape(x, (n, c, h // self.stride, self.stride, w // self.stride, self.stride))
        return self.reduce_max(x_t, (3, 5))

class MindsporeModel(nn.Cell):

    def __init__(self, config):
        super(MindsporeModel, self).__init__()

        self.premodel = config.vgg_init
        self.device = config.device_target
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, 2)
        self.conv3_128 = self.__make_layer(128, 2)
        self.conv3_256 = self.__make_layer(256, 3)
        self.conv3_512a = self.__make_layer(512, 3)
        self.conv3_512b = self.__make_layer(512, 3)
        self.max_1 = nn.MaxPool2d((2, 2), stride=(2, 2), pad_mode='same')
        self.max_4 = nn.MaxPool2d((4, 4), stride=(4, 4), pad_mode='same')
        self.max_8 = nn.MaxPool2d((8, 8), stride=(8, 8), pad_mode='same')
        self.max_16 = MyMaxPool(16)
        self.max_32 = MyMaxPool(32)

        self.upSampleLike = upSampleLike()

        # edge detection module
        self.salConv6 = nn.Conv2d(kernel_size=(5, 5), in_channels=512, out_channels=512, stride=(1, 1), dilation=(1, 1),
                                  padding=(2, 2, 2, 2), pad_mode='pad', group=1, has_bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.salConv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512, stride=(1, 1), dilation=(1, 1),
                                  padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.edgeConv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512, stride=(1, 1),
                                   dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.salConv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=256, stride=(1, 1), dilation=(1, 1),
                                  padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.salConv4_1 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(1, 1),
                                    dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.edgeConv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=256, stride=(1, 1),
                                   dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.edgeConv4_1 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(1, 1),
                                     dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.salConv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(1, 1), dilation=(1, 1),
                                  padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.edgeConv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(1, 1),
                                   dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.salConv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, stride=(1, 1), dilation=(1, 1),
                                  padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.edgeConv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, stride=(1, 1),
                                   dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.salConv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64, stride=(1, 1), dilation=(1, 1),
                                  padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.edgeConv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64, stride=(1, 1), dilation=(1, 1),
                                   padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)

        # saliency + edge + attention
        self.Reshape = P.Reshape()
        self.Transpose = P.Transpose()
        self.Tile = P.Tile()
        self.Mul = P.Mul()
        self.Add = P.Add()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 1, (1, 1))
        self.conv1_512 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=1, stride=(1, 1), dilation=(1, 1),
                                   padding=0, pad_mode='valid', group=1, has_bias=True)
        self.conv1_256 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1, stride=(1, 1), dilation=(1, 1),
                                   padding=0, pad_mode='valid', group=1, has_bias=True)
        self.conv1_128 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=1, stride=(1, 1), dilation=(1, 1),
                                   padding=0, pad_mode='valid', group=1, has_bias=True)
        self.conv1_64 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=1, stride=(1, 1), dilation=(1, 1),
                                  padding=0, pad_mode='valid', group=1, has_bias=True)
        self.conv1_32 = nn.Conv2d(kernel_size=(1, 1), in_channels=32, out_channels=1, stride=(1, 1), dilation=(1, 1),
                                  padding=0, pad_mode='valid', group=1, has_bias=True)
        self.conv3_513 = nn.Conv2d(kernel_size=(3, 3), in_channels=513, out_channels=256, stride=(1, 1),
                                   dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv3_257 = nn.Conv2d(kernel_size=(3, 3), in_channels=257, out_channels=256, stride=(1, 1),
                                   dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv257_128 = nn.Conv2d(kernel_size=(3, 3), in_channels=257, out_channels=128, stride=(1, 1),
                                     dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.dilation_256 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1, stride=(1, 1),
                                      dilation=(3, 3), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.dilation_128 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=1, stride=(1, 1),
                                      dilation=(3, 3), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.dilation_64 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=1, stride=(1, 1), dilation=(3, 3),
                                     padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.dilation_32 = nn.Conv2d(kernel_size=(1, 1), in_channels=32, out_channels=1, stride=(1, 1), dilation=(3, 3),
                                     padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)

        # conv function
        self.conv_128_1 = nn.Conv2d(128, 1, (1, 1))
        self.conv_64_1 = nn.Conv2d(64, 1, (1, 1))
        self.conv_32_1 = nn.Conv2d(32, 1, (1, 1))
        self.conv_258_128 = nn.Conv2d(kernel_size=(3, 3), in_channels=258, out_channels=128, stride=(1, 1),
                                      dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_129_128 = nn.Conv2d(kernel_size=(3, 3), in_channels=129, out_channels=128, stride=(1, 1),
                                      dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_259_128 = nn.Conv2d(kernel_size=(3, 3), in_channels=259, out_channels=128, stride=(1, 1),
                                      dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_131_64 = nn.Conv2d(kernel_size=(3, 3), in_channels=131, out_channels=64, stride=(1, 1),
                                     dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_132_64 = nn.Conv2d(kernel_size=(3, 3), in_channels=132, out_channels=64, stride=(1, 1),
                                     dilation=(1, 1), padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_65_64 = nn.Conv2d(kernel_size=(3, 3), in_channels=65, out_channels=64, stride=(1, 1), dilation=(1, 1),
                                    padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_68_32 = nn.Conv2d(kernel_size=(3, 3), in_channels=68, out_channels=32, stride=(1, 1), dilation=(1, 1),
                                    padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_69_32 = nn.Conv2d(kernel_size=(3, 3), in_channels=69, out_channels=32, stride=(1, 1), dilation=(1, 1),
                                    padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.conv_33_32 = nn.Conv2d(kernel_size=(3, 3), in_channels=33, out_channels=32, stride=(1, 1), dilation=(1, 1),
                                    padding=(1, 1, 1, 1), pad_mode='pad', group=1, has_bias=True)
        self.vgg = VGG()
        if self.device == "Ascend":
            self.init_vgg()
    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            i = i + 1
            layers.append(nn.Conv2d(in_channels=self.in_channels,
                                    out_channels=channels,
                                    kernel_size=3,
                                    stride=(1, 1),
                                    dilation=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode='pad',
                                    group=1,
                                    has_bias=False))  # same padding
            layers.append(nn.ReLU())
            self.in_channels = channels
            i = i - 1
        return nn.SequentialCell(*layers)

    def mymax_16(self, target):
        target_shape = target.shape
        resize = P.ResizeBilinear((target_shape[-2]/16, target_shape[-1]/16))
        target = resize(target)
        return target

    def mymax_32(self, target):
        target_shape = target.shape
        resize = P.ResizeBilinear((target_shape[-2]/32, target_shape[-1]/32))
        target = resize(target)
        return target

    def attention5(self, sal_5):
        att_5a = self.sigmoid(self.conv1_256(sal_5))
        att_5a = self.softmax(self.flatten(att_5a))
        att_5b = self.sigmoid(self.dilation_256(self.max_1(sal_5)))
        att_5b = self.upSampleLike(att_5b, sal_5)
        att_5b = self.softmax(self.flatten(att_5b))

        att_5 = (att_5a + att_5b) / 2.0  # (2, 1*14*14)->(256, 2, 1*14*14)

        att_5 = self.Tile(att_5, (256, 1, 1))  # (batchsize,14*14)->(256,batchsize,14*14)
        att_5 = self.Transpose(att_5, (1, 0, 2))  # (2, 256, 14*14)
        att_5 = self.Reshape(att_5, (-1, 256, 14, 14))
        att_5 = self.Mul(att_5, sal_5)
        sal_5 = self.Add(att_5, sal_5)  # (2, 256, 14, 14)
        return sal_5

    def attention4(self, sal_4):  # sal_4:2, 128, 28, 28
        att_4a = self.sigmoid(self.conv_128_1(sal_4))  # (2, 1, 28, 28)
        att_4a = self.softmax(self.flatten(att_4a))  # (2, 1*28*28)

        att_4b = self.sigmoid(self.dilation_128(self.max_1(sal_4)))  # 2, 1, 14, 14
        att_4b = self.upSampleLike(att_4b, sal_4)  # 2, 1, 28, 28
        att_4b = self.softmax(self.flatten(att_4b))  # (2, 1*28*28)

        att_4c = self.sigmoid(self.dilation_128(self.max_4(sal_4)))
        att_4c = self.upSampleLike(att_4c, sal_4)
        att_4c = self.softmax(self.flatten(att_4c))  # (2, 1*28*28)

        att_4 = (att_4a + att_4b + att_4c) / 3.0

        att_4 = self.Tile(att_4, (128, 1, 1))  # (128, 2, 1*28*28)
        att_4 = self.Transpose(att_4, (1, 0, 2))  # (2, 128, 1*28*28)
        att_4 = self.Reshape(att_4, (-1, 128, 28, 28))
        att_4 = self.Mul(att_4, sal_4)
        sal_4 = self.Add(att_4, sal_4)
        return sal_4

    def attention3(self, sal_3):  # sal_3:2, 128, 56, 56
        att_3a = self.sigmoid(self.conv_128_1(sal_3))  # (2, 1, 56, 56)
        att_3a = self.softmax(self.flatten(att_3a))  # (2, 1*56*56)

        att_3b = self.sigmoid(self.dilation_128(self.max_1(sal_3)))  # 2, 1, 28, 28
        att_3b = self.upSampleLike(att_3b, sal_3)  # 2, 1, 56, 56
        att_3b = self.softmax(self.flatten(att_3b))  # (2, 1*56*56)

        att_3c = self.sigmoid(self.dilation_128(self.max_4(sal_3)))
        att_3c = self.upSampleLike(att_3c, sal_3)
        att_3c = self.softmax(self.flatten(att_3c))  # (2, 1*56*56)

        att_3d = self.sigmoid(self.dilation_128(self.max_8(sal_3)))  # 2, 1, 7, 7
        att_3d = self.upSampleLike(att_3d, sal_3)
        att_3d = self.softmax(self.flatten(att_3d))  # (2, 1*56*56)

        att_3 = (att_3a + att_3b + att_3c + att_3d) / 4.0

        att_3 = self.Tile(att_3, (128, 1, 1))  # (128, 2, 1*56*56)
        att_3 = self.Transpose(att_3, (1, 0, 2))  # (2, 128, 1*56*56)
        att_3 = self.Reshape(att_3, (-1, 128, 56, 56))
        att_3 = self.Mul(att_3, sal_3)
        sal_3 = self.Add(att_3, sal_3)
        return sal_3

    def attention2(self, sal_2):  # (64, 112, 112)
        att_2a = self.sigmoid(self.conv_64_1(sal_2))  # (2, 1, 112, 112)
        att_2a = self.softmax(self.flatten(att_2a))  # (2, 1*112*112)

        att_2b = self.sigmoid(self.dilation_64(self.max_1(sal_2)))  # 2, 1, 56, 56
        att_2b = self.upSampleLike(att_2b, sal_2)  # 2, 1, 112, 112
        att_2b = self.softmax(self.flatten(att_2b))  # (2, 1*112*112)

        att_2c = self.sigmoid(self.dilation_64(self.max_4(sal_2)))  # 2, 1, 28, 28
        att_2c = self.upSampleLike(att_2c, sal_2)
        att_2c = self.softmax(self.flatten(att_2c))  # (2, 1*112*112)

        att_2d = self.sigmoid(self.dilation_64(self.max_8(sal_2)))  # 2, 1, 14, 14
        att_2d = self.upSampleLike(att_2d, sal_2)
        att_2d = self.softmax(self.flatten(att_2d))  # (2, 1*112*112)

        att_2e = self.sigmoid(self.dilation_64(self.max_16(sal_2)))
        att_2e = self.upSampleLike(att_2e, sal_2)
        att_2e = self.softmax(self.flatten(att_2e))

        att_2 = (att_2a + att_2b + att_2c + att_2d + att_2e) / 5.0

        att_2 = self.Tile(att_2, (64, 1, 1))  # (64, 2, 1*112*112)
        att_2 = self.Transpose(att_2, (1, 0, 2))  # (2, 64, 1*112*112)
        att_2 = self.Reshape(att_2, (-1, 64, 112, 112))
        att_2 = self.Mul(att_2, sal_2)
        sal_2 = self.Add(att_2, sal_2)
        return sal_2

    def attention1(self, sal_1):  # 2, 32, 224, 224
        att_1a = self.sigmoid(self.conv_32_1(sal_1))  # (2, 1, 224, 224)
        att_1a = self.softmax(self.flatten(att_1a))  # (2, 1*224*224)

        att_1b = self.sigmoid(self.dilation_32(self.max_1(sal_1)))  # 2, 1, 112, 112
        att_1b = self.upSampleLike(att_1b, sal_1)  # 2, 1, 224, 224
        att_1b = self.softmax(self.flatten(att_1b))  # (2, 1*224*224)

        att_1c = self.sigmoid(self.dilation_32(self.max_4(sal_1)))
        att_1c = self.upSampleLike(att_1c, sal_1)
        att_1c = self.softmax(self.flatten(att_1c))  # (2, 1*112*112)

        att_1d = self.sigmoid(self.dilation_32(self.max_8(sal_1)))  # 2, 1, 14, 14
        att_1d = self.upSampleLike(att_1d, sal_1)
        att_1d = self.softmax(self.flatten(att_1d))  # (2, 1*112*112)

        att_1e = self.sigmoid(self.dilation_32(self.max_16(sal_1)))
        att_1e = self.upSampleLike(att_1e, sal_1)
        att_1e = self.softmax(self.flatten(att_1e))

        att_1f = self.sigmoid(self.dilation_32(self.max_32(sal_1)))
        att_1f = self.upSampleLike(att_1f, sal_1)
        att_1f = self.softmax(self.flatten(att_1f))

        att_1 = (att_1a + att_1b + att_1c + att_1d + att_1e + att_1f) / 6.0

        att_1 = self.Tile(att_1, (32, 1, 1))  # (32, 2, 1*224*224)
        att_1 = self.Transpose(att_1, (1, 0, 2))  # (2, 32, 1*224*224)
        att_1 = self.Reshape(att_1, (-1, 32, 224, 224))
        att_1 = self.Mul(att_1, sal_1)
        sal_1 = self.Add(att_1, sal_1)
        return sal_1

    def init_vgg(self):
        if os.path.exists(self.premodel):
            param_dict = load_checkpoint(self.premodel)
            new_param = {}
            for key in param_dict.keys():
                new_key = "vgg.model." + key
                new_param[new_key] = param_dict[key]
            load_param_into_net(self.vgg, new_param)
            print("successfully load vgg model")
        return 0

    def construct(self, x):
        # vgg16
        if self.device == "Ascend":
            x1, x2, x3, x4, x5, x6 = self.vgg(x)
        else:
            x1 = self.conv3_64(x)  # x1:(64, 224, 224)
            x1_max = self.max_1(x1)  # x1_max:(64, 112, 112)
            x2 = self.conv3_128(x1_max)  # x2:(128, 112, 112)
            x2_max = self.max_1(x2)  # x2_max:(128, 56, 56)
            x3 = self.conv3_256(x2_max)  # x3:(256, 56, 56)
            x3_max = self.max_1(x3)  # x3_max:(256, 28, 28)
            x4 = self.conv3_512a(x3_max)  # x4:(512, 28, 28)
            x4_max = self.max_1(x4)  # x4_max:(512, 14, 14)
            x5 = self.conv3_512b(x4_max)  # x5:(512, 14, 14)
            x6 = self.max_1(x5)  # x6:(512, 7, 7)

        # sal_conv
        sal_6 = self.relu(self.salConv6(x6))
        sal_6 = self.sigmoid(self.salConv6(sal_6))  # sal_6:(512, 7, 7)

        sal_5 = self.relu(self.salConv5(x5))
        sal_5 = self.sigmoid(self.salConv5(sal_5))  # sal_5:(512, 14, 14)

        sal_4 = self.relu(self.salConv4(x4))
        sal_4 = self.sigmoid(self.salConv4_1(sal_4))  # sal_4:(256, 28, 28)

        sal_3 = self.relu(self.salConv3(x3))
        sal_3 = self.sigmoid(self.salConv3(sal_3))  # sal_3:(256, 56, 56)

        sal_2 = self.relu(self.salConv2(x2))
        sal_2 = self.sigmoid(self.salConv2(sal_2))  # sal_2:(128, 112, 112)

        sal_1 = self.relu(self.salConv1(x1))
        sal_1 = self.sigmoid(self.salConv1(sal_1))  # sal_1:(64, 224, 224)

        # edge_conv
        edg_5 = self.relu(self.edgeConv5(x5))
        edg_5 = self.sigmoid(self.edgeConv5(edg_5))  # edg_5:(512, 14, 14)

        edg_4 = self.relu(self.edgeConv4(x4))
        edg_4 = self.sigmoid(self.edgeConv4_1(edg_4))  # edg_4:(256, 28, 28)

        edg_3 = self.relu(self.edgeConv3(x3))
        edg_3 = self.sigmoid(self.edgeConv3(edg_3))  # edg_3:(256, 56, 56)

        edg_2 = self.relu(self.edgeConv2(x2))
        edg_2 = self.sigmoid(self.edgeConv2(edg_2))  # edg_2:(128, 112, 112)

        edg_1 = self.relu(self.edgeConv1(x1))
        edg_1 = self.sigmoid(self.edgeConv1(edg_1))  # edg_1:(64, 224, 224)  sigmoid-88

        # saliency from sal_6 sal_6_up
        saliency6 = self.sigmoid(self.conv1_512(sal_6))  # saliency6_up:sigmoid((1, 7, 7))
        saliency6_up = self.upSampleLike(saliency6, x1)  # saliency6_up:(1, 224, 224)

        # saliency from sal_5 sal_5_up edge5 edge5_up
        edge5 = self.sigmoid(self.conv1_512(edg_5))  # edge5:(1, 14, 14) sigmoid-92
        edge5_up = self.upSampleLike(edge5, x1)  # edge5_up:(2, 1, 224, 224)

        sal_5 = P.Concat(axis=1)([sal_5, self.upSampleLike(saliency6, sal_5)])
        sal_5 = self.sigmoid(self.conv3_513(sal_5))  # sal_5: 256, 14, 14  sigmoid-94

        sal_5 = P.Concat(axis=1)([self.attention5(sal_5), edge5])  # sal_5:(257, 14, 14)
        sal_5 = self.sigmoid(self.conv3_257(sal_5))  # sal_5:(256, 14, 14)
        saliency5 = self.sigmoid(self.conv1_256(sal_5))  # saliency:(1, 14, 14)   sigmoid-107
        sal_5_up = self.upSampleLike(saliency5, x1)  # sal_5_up:(1, 224, 224)

        # saliency from sal_4 sal_4_up edge4 edge4_up
        edg_4 = P.Concat(axis=1)([edg_4, self.upSampleLike(edge5, sal_4)])  # (257, 28, 28)
        edg_4 = self.sigmoid(self.conv257_128(edg_4))  # (128, 28, 28)  sigmoid-109
        edge4 = self.sigmoid(self.conv1_128(edg_4))  # edge5:(1, 28, 28)  sigmoid-111
        edge4_up = self.upSampleLike(edge4, x1)  # edge5_up:(2, 1, 224, 224)

        sal_4 = P.Concat(axis=1)(
            [sal_4, self.upSampleLike(saliency6, sal_4), self.upSampleLike(saliency5, sal_4)])  # (258, 28, 28)
        sal_4 = self.sigmoid(self.conv_258_128(sal_4))  # sigmoid-112

        sal_4 = P.Concat(axis=1)([self.attention4(sal_4), edge4])
        sal_4 = self.sigmoid(self.conv_129_128(sal_4))  # sigmoid-126

        saliency4 = self.sigmoid(self.conv1_128(sal_4))  # sigmoid-128
        sal_4_up = self.upSampleLike(saliency4, x1)

        # saliency from sal_3 sal_3_up edge3 edge3_up
        edg_3 = P.Concat(axis=1)(
            [edg_3, self.upSampleLike(edge5, sal_3), self.upSampleLike(edge4, sal_3)])  # (258, 56, 56)
        edg_3 = self.sigmoid(self.conv_258_128(edg_3))  # (128, 56, 56)
        edge3 = self.sigmoid(self.conv1_128(edg_3))
        edge3_up = self.upSampleLike(edge3, x1)

        sal_3 = P.Concat(axis=1)([sal_3, self.upSampleLike(saliency6, sal_3), self.upSampleLike(saliency5, sal_3),
                                  self.upSampleLike(saliency4, sal_3)])
        sal_3 = self.sigmoid(self.conv_259_128(sal_3))  # 2, 128, 56, 56

        sal_3 = P.Concat(axis=1)([self.attention3(sal_3), edge3])
        sal_3 = self.sigmoid(self.conv_129_128(sal_3))  # sigmoid-151

        saliency3 = self.sigmoid(self.conv1_128(sal_3))  # sigmoid-153
        sal_3_up = self.upSampleLike(saliency3, x1)

        # saliency from sal_2 sal_2_up edge2 edge2_up
        edg_2 = P.Concat(axis=1)([edg_2, self.upSampleLike(edge5, edg_2), self.upSampleLike(edge4, edg_2),
                                  self.upSampleLike(edge3, edg_2)])  # (131, 112, 112)
        edg_2 = self.sigmoid(self.conv_131_64(edg_2))  # (64, 112, 112)

        edge2 = self.sigmoid(self.conv1_64(edg_2))
        edge2_up = self.upSampleLike(edge2, x1)

        sal_2 = P.Concat(axis=1)([sal_2, self.upSampleLike(saliency6, sal_2), self.upSampleLike(saliency5, sal_2),
                                  self.upSampleLike(saliency4, sal_2), self.upSampleLike(saliency3, sal_2)])
        sal_2 = self.sigmoid(self.conv_132_64(sal_2))  # (2, 64, 112, 112)

        sal_2 = P.Concat(axis=1)([self.attention2(sal_2), edge2])
        sal_2 = self.sigmoid(self.conv_65_64(sal_2))

        saliency2 = self.sigmoid(self.conv1_64(sal_2))
        sal_2_up = self.upSampleLike(saliency2, x1)

        # saliency from sal_1 sal_1_up edge1 edge1_up
        edg_1 = P.Concat(axis=1)(
            [edg_1, self.upSampleLike(edge5, sal_1), self.upSampleLike(edge4, sal_1), self.upSampleLike(edge3, sal_1),
             self.upSampleLike(edge2, sal_1)])  # (68,224,224)
        edg_1 = self.sigmoid(self.conv_68_32(edg_1))  # (32, 224, 224)
        edge1 = self.sigmoid(self.conv1_32(edg_1))

        sal_1 = P.Concat(axis=1)([sal_1, self.upSampleLike(saliency6, sal_1), self.upSampleLike(saliency5, sal_1),
                                  self.upSampleLike(saliency4, sal_1), self.upSampleLike(saliency3, sal_1),
                                  self.upSampleLike(saliency2, sal_1)])
        sal_1 = self.sigmoid(self.conv_69_32(sal_1))  # 32, 224, 224

        sal_1 = P.Concat(axis=1)([self.attention1(sal_1), edge1])
        sal_1 = self.sigmoid(self.conv_33_32(sal_1))

        saliency1 = self.sigmoid(self.conv1_32(sal_1))  # (1, 224, 224)

        return [saliency6_up, edge5_up, sal_5_up, edge4_up, sal_4_up, edge3_up, sal_3_up, edge2_up, sal_2_up, saliency1,
                edge1]


if __name__ == "__main__":
    m = MindsporeModel()
    print(m)
