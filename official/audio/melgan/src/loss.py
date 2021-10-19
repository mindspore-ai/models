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
"""loss definition"""
import mindspore.nn as nn
from mindspore.ops import operations as P


class GeneratorLoss_score(nn.Cell):
    """Generator loss"""
    def __init__(self):
        super(GeneratorLoss_score, self).__init__()
        self.pow = P.Pow()
        self.sum = P.ReduceSum(False)
        self.mean = P.ReduceMean(False)

    def construct(self, x6):
        loss = self.pow(x6 - 1, 2.0)
        loss = self.sum(loss, (1, 2))
        loss = self.mean(loss)
        return loss


class GeneratorLoss_fmap(nn.Cell):
    def __init__(self):
        super(GeneratorLoss_fmap, self).__init__()
        self.mean = P.ReduceMean(False)
        self.abs = P.Abs()

    def construct(self, feat_f, feat_r):
        loss1 = self.abs(feat_f - feat_r)
        loss1 = self.mean(loss1)
        return 10 * loss1

class MelganLoss_G(nn.Cell):
    """melgan loss of generator"""
    def __init__(self):
        super(MelganLoss_G, self).__init__()

        self.criterion1 = GeneratorLoss_score()
        self.criterion2 = GeneratorLoss_fmap()

    def construct(self, output1, output2):
        loss = self.criterion1(output1[0][6])
        loss = self.criterion1(output1[1][6]) + loss
        loss = self.criterion1(output1[2][6]) + loss
        for i in range(3):
            for j in range(6):
                loss = loss + self.criterion2(output1[i][j], output2[i][j])

        return loss


class DiscriminatorLoss(nn.Cell):
    """discriminator loss"""
    def __init__(self, num):
        super(DiscriminatorLoss, self).__init__()
        self.pow = P.Pow()
        self.sum = P.ReduceSum(False)
        self.mean = P.ReduceMean(False)
        self.mus = num

    def construct(self, x6):
        loss = self.pow(x6 - self.mus, 2.0)
        loss = self.sum(loss, (1, 2))
        loss = self.mean(loss)
        return loss


class MelganLoss_D(nn.Cell):
    """melgan loss of discriminator"""
    def __init__(self):
        super(MelganLoss_D, self).__init__()

        self.criterion1 = DiscriminatorLoss(0)  # fake
        self.criterion2 = DiscriminatorLoss(1)  # real

    def construct(self, output1, output2):
        loss = self.criterion1(output1[0][6])
        loss = self.criterion1(output1[1][6]) + loss
        loss = self.criterion1(output1[2][6]) + loss
        loss = self.criterion2(output2[0][6]) + loss
        loss = self.criterion2(output2[1][6]) + loss
        loss = self.criterion2(output2[2][6]) + loss
        return loss
