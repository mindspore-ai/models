#!/bin/bash
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

import math
import gin

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

ngates = 2

def ginM(n):
    return gin.query_parameter(f'%{n}')
gin.external_configurable(nn.MaxPool2d, module='nn')
gin.external_configurable(ops.ResizeBilinear, module='nn')


class LN(nn.Cell):
    def construct(self, x):
        layer_norm = nn.LayerNorm(x.shape[1:], 1, 1)
        return layer_norm(x)

@gin.configurable
def pCnv(inp, out, use_batch_statistics, groups=1):
    return nn.SequentialCell([
        nn.Conv2d(inp, out, 1, has_bias=False, group=groups),
        nn.BatchNorm2d(out, affine=True, use_batch_statistics=use_batch_statistics)
    ])

@gin.configurable
def dsCnv(inp, k, use_batch_statistics):
    return nn.SequentialCell([
        nn.Conv2d(inp, inp, k, group=inp, has_bias=False, pad_mode="pad", padding=(k - 1) // 2),
        nn.BatchNorm2d(inp, affine=True, use_batch_statistics=use_batch_statistics)
    ])


def basicConv(inp, out, kernel_size=1, padding=0):
    return nn.SequentialCell([
        pCnv(inp, out),
        dsCnv(out, kernel_size)
    ])


class InceptionA(nn.Cell):
    def __init__(self, in_channels, f1=64, f2=64, f3=96, f4=32):
        super(InceptionA, self).__init__()
        self.branch1x1 = basicConv(in_channels, f1, kernel_size=1)
        self.branch5x5_1 = basicConv(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = basicConv(48, f2, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = basicConv(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = basicConv(64, f3, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = basicConv(f3, f3, kernel_size=3, padding=1)
        self.branch_pool = basicConv(in_channels, f4, kernel_size=1)
        self.concat = mindspore.ops.Concat(1)

    def construct(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        x = branch_pool(x)
        branch_pool = self.branch_pool(x)

        outputs = (branch1x1, branch5x5, branch3x3dbl, branch_pool)
        return self.concat(outputs)

class Gate(nn.Cell):
    def __init__(self, ifsz):
        super().__init__()

    def construct(self, x):
        t0, t1 = mnp.split(x, ngates, axis=1)
        tanh = nn.Tanh()
        t0 = tanh(t0)
        sub = ops.Sub()
        t1 = sub(t1, 2)
        sigmoid = nn.Sigmoid()
        t1 = sigmoid(t1)

        return t1*t0


def customGC(module):
    def custom_forward(*inputs):
        inputs = module(inputs[0])
        return inputs
    return custom_forward


@gin.configurable
class GateBlock(nn.Cell):
    def __init__(self, ifsz, ofsz, gt=True, ksz=3, GradCheck=gin.REQUIRED):
        super(GateBlock, self).__init__()

        cfsz = int(math.floor(ifsz/2))
        oc = cfsz*ngates

        self.sq = nn.SequentialCell([
            pCnv(ifsz, cfsz),
            nn.ELU(),
            pCnv(cfsz, cfsz*ngates),
            dsCnv(cfsz*ngates, ksz),
            Gate(cfsz),
            pCnv(cfsz, ifsz),
            nn.ELU()
        ])

        if oc == 256:
            self.sq = nn.SequentialCell([
                pCnv(ifsz, cfsz),
                nn.ELU(),
                InceptionA(cfsz),
                Gate(cfsz),
                pCnv(cfsz, ifsz),
                nn.ELU()
            ])

        if oc == 512:
            self.sq = nn.SequentialCell([
                pCnv(ifsz, cfsz),
                nn.ELU(),
                InceptionA(cfsz, 64*2, 64*2, 6*2, 32*2),
                Gate(cfsz),
                pCnv(cfsz, ifsz),
                nn.ELU()
            ])

        self.gt = gt
        self.oc = oc
        self.gc = GradCheck
    def construct(self, x):
        y = self.sq(x)
        out = x + y
        return out

@gin.configurable
class InitBlock(nn.Cell):
    def __init__(self, fup, n_channels):
        super().__init__()
        self.n1 = LN()
        self.Initsq = nn.SequentialCell([
            pCnv(n_channels, fup),
            nn.Softmax(axis=1),
            dsCnv(fup, 11),
            LN()
        ])

    def construct(self, x):
        x = self.n1(x)
        xt = x
        x = self.Initsq(x)
        x = mnp.concatenate((x, xt), 1)
        return x


@gin.configurable
class OrigamiNet(nn.Cell):
    def __init__(self, n_channels, o_classes, wmul, lreszs, lszs, nlyrs, fup, GradCheck, reduceAxis=3):
        super().__init__()
        self.localization = nn.SequentialCell([
            nn.Conv2d(n_channels, 8, kernel_size=13, has_bias=True),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=9, has_bias=True),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=7, has_bias=True),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        ])

        self.fc_loc = nn.SequentialCell([
            nn.Dense(10 * 6 * 6, 32),
            nn.ReLU(),
            nn.Dense(32, 1)
        ])

        self.lreszs = lreszs
        self.Initsq = InitBlock(fup)
        layers = []
        isz = fup + n_channels
        osz = isz
        for i in range(nlyrs):
            osz = int(math.floor(lszs[i] * wmul)) if i in lszs else isz
            layers.append(GateBlock(isz, osz, True, 3))

            if isz != osz:
                pcnv = pCnv(isz, osz)
                layers.append(pcnv)
                elu = nn.ELU()
                layers.append(elu)
            isz = osz

            if i in lreszs:
                layers.append(lreszs[i])

        layers.append(LN())

        self.Gatesq = nn.SequentialCell(layers)
        self.Finsq = nn.SequentialCell([
            pCnv(osz, o_classes),
            nn.ELU()
        ])

        self.n1 = LN()
        self.it = 0
        self.gc = GradCheck
        self.reduceAxis = reduceAxis

    def construct(self, x):
        x = self.Initsq(x)
        x = self.Gatesq(x)
        x = self.Finsq(x)
        x = self.n1(x)
        x = mnp.reshape(x, (x.shape[0], x.shape[1], -1))
        transpose = ops.Transpose()
        x = transpose(x, (0, 2, 1))

        return x

class Reshape(nn.Cell):
    def __init__(self, h, w):
        super(Reshape, self).__init__()
        self.h = h
        self.w = w

    def construct(self, x):
        return mnp.reshape(x, (x.shape[0], x.shape[1], self.h, self.w))
