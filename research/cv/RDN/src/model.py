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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import Parameter


class PixelShuffle(nn.Cell):
    def __init__(self, upscale_factor):
        super().__init__()
        self.DepthToSpace = ops.DepthToSpace(upscale_factor)

    def construct(self, x):
        return self.DepthToSpace(x)


class RDB_Conv(nn.Cell):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.SequentialCell([
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad"),
            nn.ReLU()
        ])

    def construct(self, x):
        out = self.conv(x)
        op = ops.Concat(1)
        return op((x, out))


class RDB(nn.Cell):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.SequentialCell(convs)
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1, pad_mode="pad")  # need test

    def construct(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Cell):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad")
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad")

        # Residual dense blocks and dense feature fusion
        self.RDBs = nn.CellList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.SequentialCell([
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1, pad_mode="pad"),  # need test
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad")
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.SequentialCell([
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad"),
                PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad")
            ])
        elif r == 4:
            self.UPNet = nn.SequentialCell([
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad"),
                PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad"),
                PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1, pad_mode="pad")
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def construct(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = ()
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out += (x,)
        op = ops.Concat(1)
        x = self.GFF(op(RDBs_out))
        x += f__1
        return self.UPNet(x)

