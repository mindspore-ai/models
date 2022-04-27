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
'''hed model'''
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.ops import operations as P

def make_bilinear_weights(size, num_channels):
    '''make bilinear weights'''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = Tensor.from_numpy(filt)
    w = np.zeros((num_channels, num_channels, size, size), np.float32)
    w = Tensor.from_numpy(w)
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return mnp.array(w)

def prepare_aligned_crop():
    """ Prepare for aligned crop. """
    # Re-implement the logic in deploy.prototxt and
    #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
    # Other reference materials:
    #   hed/include/caffe/layer.hpp
    #   hed/include/caffe/vision_layers.hpp
    #   hed/include/caffe/util/coords.hpp
    #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

    def map_inv(m):
        """ Mapping inverse. """
        a, b = m
        return 1 / a, -b / a

    def map_compose(m1, m2):
        """ Mapping compose. """
        a1, b1 = m1
        a2, b2 = m2
        return a1 * a2, a1 * b2 + b1

    def deconv_map(kernel_h, stride_h, pad_h):
        """ Deconvolution coordinates mapping. """
        return stride_h, (kernel_h - 1) / 2 - pad_h

    def conv_map(kernel_h, stride_h, pad_h):
        """ Convolution coordinates mapping. """
        return map_inv(deconv_map(kernel_h, stride_h, pad_h))

    def pool_map(kernel_h, stride_h, pad_h):
        """ Pooling coordinates mapping. """
        return conv_map(kernel_h, stride_h, pad_h)

    x_map = (1, 0)
    conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
    conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
    pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

    conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
    conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
    pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

    conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
    conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
    conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
    pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

    conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
    conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
    conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
    pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

    conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
    conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
    conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

    score_dsn1_map = conv1_2_map
    score_dsn2_map = conv2_2_map
    score_dsn3_map = conv3_3_map
    score_dsn4_map = conv4_3_map
    score_dsn5_map = conv5_3_map

    upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
    upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
    upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
    upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

    crop1_margin = int(score_dsn1_map[1])
    crop2_margin = int(upsample2_map[1])
    crop3_margin = int(upsample3_map[1])
    crop4_margin = int(upsample4_map[1])
    crop5_margin = int(upsample5_map[1])

    return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

class HED(nn.Cell):
    '''HED'''
    def __init__(self):
        super(HED, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35, pad_mode="pad", has_bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, pad_mode="pad", has_bias=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, pad_mode="pad", has_bias=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, pad_mode="pad", has_bias=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, pad_mode="pad", has_bias=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, pad_mode="pad", has_bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, pad_mode="pad", has_bias=True)

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.score_dsn1 = nn.Conv2d(64, 1, 1, has_bias=True)
        self.score_dsn2 = nn.Conv2d(128, 1, 1, has_bias=True)
        self.score_dsn3 = nn.Conv2d(256, 1, 1, has_bias=True)
        self.score_dsn4 = nn.Conv2d(512, 1, 1, has_bias=True)
        self.score_dsn5 = nn.Conv2d(512, 1, 1, has_bias=True)
        self.score_final = nn.Conv2d(5, 1, 1, has_bias=True)

        self.crop1_margin, self.crop2_margin,\
        self.crop3_margin, self.crop4_margin,\
        self.crop5_margin = prepare_aligned_crop()

        self.weight_deconv2 = make_bilinear_weights(4, 1)
        self.weight_deconv3 = make_bilinear_weights(8, 1)
        self.weight_deconv4 = make_bilinear_weights(16, 1)
        self.weight_deconv5 = make_bilinear_weights(32, 1)

        self.conv_transpose2 = nn.Conv2dTranspose(1, 1, 4, weight_init=self.weight_deconv2, stride=2)
        self.conv_transpose3 = nn.Conv2dTranspose(1, 1, 8, weight_init=self.weight_deconv3, stride=4)
        self.conv_transpose4 = nn.Conv2dTranspose(1, 1, 16, weight_init=self.weight_deconv4, stride=8)
        self.conv_transpose5 = nn.Conv2dTranspose(1, 1, 32, weight_init=self.weight_deconv5, stride=16)

    def construct(self, x):
        '''vgg'''
        conv1_1 = self.conv1_1(x)

        conv1_1 = self.relu(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)

        conv1_2 = self.relu(conv1_2)
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        so1 = self.score_dsn1(conv1_2)
        so2 = self.score_dsn2(conv2_2)
        so3 = self.score_dsn3(conv3_3)
        so4 = self.score_dsn4(conv4_3)
        so5 = self.score_dsn5(conv5_3)

        upsample2 = self.conv_transpose2(so2)
        upsample3 = self.conv_transpose3(so3)
        upsample4 = self.conv_transpose4(so4)
        upsample5 = self.conv_transpose5(so5)

        image_w = 321
        image_h = 481
        so1 = so1[:, :, self.crop1_margin:self.crop1_margin + image_w,
                  self.crop1_margin:self.crop1_margin + image_h]
        so2 = upsample2[:, :, self.crop2_margin:self.crop2_margin + image_w,
                        self.crop2_margin:self.crop2_margin + image_h]
        so3 = upsample3[:, :, self.crop3_margin:self.crop3_margin + image_w,
                        self.crop3_margin:self.crop3_margin + image_h]
        so4 = upsample4[:, :, self.crop4_margin:self.crop4_margin + image_w,
                        self.crop4_margin:self.crop4_margin + image_h]
        so5 = upsample5[:, :, self.crop5_margin:self.crop5_margin + image_w,
                        self.crop5_margin:self.crop5_margin + image_h]

        op = ops.Concat(1)
        fusecat = op((so1, so2, so3, so4, so5))
        fuse = self.score_final(fusecat)

        so1 = P.Sigmoid()(so1)
        so2 = P.Sigmoid()(so2)
        so3 = P.Sigmoid()(so3)
        so4 = P.Sigmoid()(so4)
        so5 = P.Sigmoid()(so5)
        fuse = P.Sigmoid()(fuse)
        return so1, so2, so3, so4, so5, fuse
