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
model architecture of densenet
"""
import math
import numpy as np
import mindspore.common.initializer
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, Parameter

def myphi(x, m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)
class AngleLinear(nn.Cell):
    def __init__(self, in_features, out_features, m=4, phiflag=False):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        norm = ops.L2Normalize(axis=0, epsilon=1e-5)
        weight = np.random.uniform(-1, 1, (in_features, out_features))
        weight = Tensor(weight, mindspore.float32)
        weight = norm(weight)
        self.weight = Parameter(weight)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
    def construct(self, inputs):
        x = inputs
        w = self.weight

        reshape = ops.Reshape()
        mul = ops.MatMul()
        tr = ops.Transpose()
        mult = ops.Mul()
        ones = ops.Ones()
        floor = ops.Floor()
        acos = ops.ACos()
        norm = ops.L2Normalize(axis=0, epsilon=1e-5)
        pows = ops.Pow()

        onesw = ones((w.shape[0], w.shape[1]), mindspore.float32)
        at = mult(onesw, w)
        ws = norm(at)
        xx = pows(x, 2).sum(axis=1)
        xx = pows(xx, 0.5)
        ww = pows(ws, 2).sum(axis=0)
        ww = pows(ww, 0.5)
        cos_theta = mul(x, ws)
        cos_theta = cos_theta / reshape(ww, (1, -1))
        cos_theta = tr(cos_theta, (1, 0))
        cos_theta = cos_theta / reshape(xx, (1, -1))

        cos_theta = tr(cos_theta, (1, 0))
        cos_m_theta = 8 * pows(cos_theta, 4) - 8 * pows(cos_theta, 2) + 1

        theta = acos(cos_theta)

        k = floor(self.m * theta / 3.14159265)
        n_one = k * 0.0 - 1
        phi_theta = pows(n_one, k) * cos_m_theta - 2 * k

        cos_theta = mult(cos_theta, reshape(xx, (-1, 1)))
        phi_theta = mult(phi_theta, reshape(xx, (-1, 1)))

        output = [cos_theta, phi_theta]
        return output



class sphere20a(nn.Cell):
    def __init__(self, classnum=10574, feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, pad_mode="pad", padding=1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, pad_mode="pad", padding=1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, pad_mode="pad", padding=1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, pad_mode="pad", padding=1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, pad_mode="pad", padding=1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, pad_mode="pad", padding=1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, pad_mode="pad", padding=1)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, pad_mode="pad", padding=1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, pad_mode="pad", padding=1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, pad_mode="pad", padding=1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, pad_mode="pad", padding=1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, pad_mode="pad", padding=1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, pad_mode="pad", padding=1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Dense(512*7*6, 512, bias_init='normal')
        if not self.feature:
            self.fc6 = AngleLinear(512, self.classnum)
        print('*'*20)
        print(self.classnum)
        print('*'*20)

    def construct(self, x):
        reshape = ops.Reshape()

        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = reshape(x, (x.shape[0], -1))

        x = self.fc5(x)

        if self.feature: return x
        x = self.fc6(x)
        return x
