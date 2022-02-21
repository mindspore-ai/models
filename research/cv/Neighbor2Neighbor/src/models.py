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
'''model'''
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np

class UpsampleCat(nn.Cell):
    '''UpsampleCat'''
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.deconv = nn.Conv2dTranspose(in_nc, out_nc, 2, 2, \
                         padding=0, has_bias=False, weight_init="HeNormal")#weight*=0.1
        self.concat = ops.Concat(axis=1)#NCHW

    def construct(self, x1, x2):
        '''construct'''
        x1 = self.deconv(x1)
        return self.concat((x1, x2))

def rotate(x, angle):
    if angle == 0:
        return x
    if angle == 90:
        return np.rot90(x, 1, (3, 2))
    if angle == 180:
        return np.rot90(x, 2, (3, 2))
    return np.rot90(x, 3, (3, 2))

def conv_func(x, conv, blindspot, pad):
    ofs = 0 if (not blindspot) else 1
    if ofs > 0:
        x = pad(x)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x

def pool_func(x, pool, blindspot, pad):
    if blindspot:
        x = pad(x[:, :, :-1, :])
    x = pool(x)
    return x

class UNet(nn.Cell):
    """
    args:
         in_nc=3,
         out_nc=3,
         n_feature=48,
         blindspot=False,
         zero_last=False
    """
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 n_feature=48,
                 blindspot=False,
                 zero_last=False):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.blindspot = blindspot
        self.zero_last = zero_last
        self.act = nn.LeakyReLU(alpha=0.2)

        # Encoder part
        self.enc_conv0 = nn.Conv2d(self.in_nc, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.enc_conv1 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc_conv2 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc_conv3 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc_conv4 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.pool4 = nn.MaxPool2d(2, 2)

        self.enc_conv5 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.pool5 = nn.MaxPool2d(2, 2)

        self.enc_conv6 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1,
                                   pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1

        # Decoder part
        self.up5 = UpsampleCat(self.n_feature, self.n_feature)
        self.dec_conv5a = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.dec_conv5b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1

        self.up4 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv4a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.dec_conv4b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1

        self.up3 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv3a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.dec_conv3b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1

        self.up2 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv2a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.dec_conv2b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1

        self.up1 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)

        # Output stages
        self.dec_conv1a = nn.Conv2d(self.n_feature * 2 + self.in_nc, 96, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.dec_conv1b = nn.Conv2d(96, 96, 3, 1,
                                    pad_mode="pad", padding=1, has_bias=True, weight_init="HeNormal")#weight*=0.1
        if blindspot:
            self.nin_a = nn.Conv2d(96 * 4, 96 * 4, 1, 1,
                                   pad_mode="pad", padding=0, has_bias=True, weight_init="HeNormal")#weight*=0.1
            self.nin_b = nn.Conv2d(96 * 4, 96, 1, 1,
                                   pad_mode="pad", padding=0, has_bias=True, weight_init="HeNormal")#weight*=0.1
        else:
            self.nin_a = nn.Conv2d(96, 96, 1, 1,
                                   pad_mode="pad", padding=0, has_bias=True, weight_init="HeNormal")#weight*=0.1
            self.nin_b = nn.Conv2d(96, 96, 1, 1,
                                   pad_mode="pad", padding=0, has_bias=True, weight_init="HeNormal")#weight*=0.1
        if self.zero_last:
            self.nin_c = nn.Conv2d(96, self.out_nc, 1, 1,
                                   pad_mode="pad", padding=0, has_bias=True)
        else:
            self.nin_c = nn.Conv2d(96, self.out_nc, 1, 1,
                                   pad_mode="pad", padding=0, has_bias=True, weight_init="HeNormal")#weight*=0.1
        self.concat0 = ops.Concat(axis=0)
        self.concat1 = ops.Concat(axis=1)
        # (padding_left, padding_right, padding_top, padding_bottom)
        #self.pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (0, 0)))
        self.split = ops.Split(axis=0, output_num=4)

    def construct(self, x):
        '''construct'''
        # Input stage
        blindspot = self.blindspot
        if blindspot:
            x = self.concat0((rotate(x, 0), rotate(x, 90), rotate(x, 180), rotate(x, 270)))
        # Encoder part
        pool0 = x
        x = self.act(conv_func(x, self.enc_conv0, blindspot, self.pad))
        x = self.act(conv_func(x, self.enc_conv1, blindspot, self.pad))
        x = pool_func(x, self.pool1, blindspot, self.pad)
        pool1 = x

        x = self.act(conv_func(x, self.enc_conv2, blindspot, self.pad))
        x = pool_func(x, self.pool2, blindspot, self.pad)
        pool2 = x

        x = self.act(conv_func(x, self.enc_conv3, blindspot, self.pad))
        x = pool_func(x, self.pool3, blindspot, self.pad)
        pool3 = x

        x = self.act(conv_func(x, self.enc_conv4, blindspot, self.pad))
        x = pool_func(x, self.pool4, blindspot, self.pad)
        pool4 = x

        x = self.act(conv_func(x, self.enc_conv5, blindspot, self.pad))
        x = pool_func(x, self.pool5, blindspot, self.pad)

        x = self.act(conv_func(x, self.enc_conv6, blindspot, self.pad))

        # Decoder part
        x = self.up5(x, pool4)
        x = self.act(conv_func(x, self.dec_conv5a, blindspot, self.pad))
        x = self.act(conv_func(x, self.dec_conv5b, blindspot, self.pad))

        x = self.up4(x, pool3)
        x = self.act(conv_func(x, self.dec_conv4a, blindspot, self.pad))
        x = self.act(conv_func(x, self.dec_conv4b, blindspot, self.pad))

        x = self.up3(x, pool2)
        x = self.act(conv_func(x, self.dec_conv3a, blindspot, self.pad))
        x = self.act(conv_func(x, self.dec_conv3b, blindspot, self.pad))

        x = self.up2(x, pool1)
        x = self.act(conv_func(x, self.dec_conv2a, blindspot, self.pad))
        x = self.act(conv_func(x, self.dec_conv2b, blindspot, self.pad))

        x = self.up1(x, pool0)

        # Output stage
        if blindspot:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot, self.pad))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot, self.pad))
            x = self.pad(x[:, :, :-1, :])
            x = self.split(x)
            x = self.concat1((rotate(x, 0), rotate(x, 270), rotate(x, 180), rotate(x, 90)))
            x = self.act(conv_func(x, self.nin_a, blindspot, self.pad))
            x = self.act(conv_func(x, self.nin_b, blindspot, self.pad))
            x = conv_func(x, self.nin_c, blindspot, self.pad)
        else:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot, self.pad))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot, self.pad))
            x = self.act(conv_func(x, self.nin_a, blindspot, self.pad))
            x = self.act(conv_func(x, self.nin_b, blindspot, self.pad))
            x = conv_func(x, self.nin_c, blindspot, self.pad)
        return x

class UNetWithLossCell(nn.Cell):
    '''UNetWithLossCell'''
    def __init__(self, network):
        super(UNetWithLossCell, self).__init__()
        self.network = network
        self.reduceSum = ops.ReduceSum(keep_dims=False)
        self.power = ops.Pow()
    def construct(self, noisy_sub1, noisy_sub2, exp_diff, Lambda):
        noisy_output = self.network(noisy_sub1)
        diff = noisy_output - noisy_sub2
        loss1 = self.power(diff, 2.0)
        loss1 = self.reduceSum(loss1)
        loss2 = Lambda * self.reduceSum(self.power((diff - exp_diff), 2.0))
        return loss1 + loss2
