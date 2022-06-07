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
"""Vision GNN (ViG)"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype

from .misc import DropPath2D, Identity, trunc_array
from .gcn_lib import MRGraphConv2d


class Grapher(nn.Cell):
    """Grapher"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_path=0.,
                 k=9, dilation=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.SequentialCell([
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(in_features),
        ])
        self.graph_conv = nn.SequentialCell([
            MRGraphConv2d(in_features, hidden_features, k=k, dilation=dilation),
            nn.BatchNorm2d(hidden_features),
            nn.GELU(),
        ])
        self.fc2 = nn.SequentialCell([
            nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(out_features),
        ])
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

    def construct(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = shortcut + self.drop_path(x)
        return x


class Mlp(nn.Cell):
    """Mlp"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.SequentialCell([
            nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.GELU(),
        ])
        self.fc2 = nn.SequentialCell([
            nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(out_features),
        ])
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

    def construct(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = shortcut + self.drop_path(x)
        return x


class Block(nn.Cell):
    """ ViG Block"""

    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, k=9, dilation=1):
        super().__init__()
        self.grapher = Grapher(dim, hidden_features=int(dim * 2), out_features=dim,
                               act_layer=act_layer, drop_path=drop_path, k=k, dilation=dilation)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), out_features=dim,
                       act_layer=act_layer, drop_path=drop_path)

    def construct(self, x):
        x = self.grapher(x)
        x = self.mlp(x)
        return x


class PatchEmbed(nn.Cell):
    """ Image to Visual Embeddings
    """

    def __init__(self, dim=768):
        super().__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=dim//8, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(dim//8),
            nn.GELU(),
        ])
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(in_channels=dim//8, out_channels=dim//4, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(dim//4),
            nn.GELU(),
        ])
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(in_channels=dim//4, out_channels=dim//2, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(dim//2),
            nn.GELU(),
        ])
        self.conv4 = nn.SequentialCell([
            nn.Conv2d(in_channels=dim//2, out_channels=dim, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        ])
        self.conv5 = nn.SequentialCell([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(dim),
        ])

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class ViG(nn.Cell):
    """ ViG (Visioin GNN)
    """

    def __init__(self, num_classes=1000, dim=768, depth=12, mlp_ratio=4., drop_path_rate=0., k=9, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        self.patch_embed = PatchEmbed(dim)

        self.pos_embed = Parameter(Tensor(trunc_array([1, dim, 14, 14]), dtype=mstype.float32),
                                   name="pos_embed")

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        num_knn = [int(x) for x in np.linspace(k, 2 * k, depth)]  # number of knn's k
        max_dilation = 196 // max(num_knn)
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim, mlp_ratio=mlp_ratio, drop_path=dpr[i], k=num_knn[i], dilation=min(max_dilation, i//4+1)))
        self.blocks = nn.CellList(blocks)

        # Classifier head
        self.head = nn.SequentialCell([
            nn.Conv2d(in_channels=dim, out_channels=1024, kernel_size=1, has_bias=True),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1, has_bias=True),
        ])

        self.init_weights()
        print("================================success================================")

    def init_weights(self):
        """init_weights"""
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    def forward_features(self, x):
        """ViG forward_features"""
        x = self.patch_embed(x) + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = P.ReduceMean(True)(x, [2, 3])
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x).squeeze(-1).squeeze(-1)
        return x


def vig_ti_patch16_224(args):
    """vig_ti_patch16_224"""
    num_classes = args.num_classes
    dim = 192
    depth = 12
    mlp_ratio = 4
    drop_path_rate = args.drop_path_rate
    model = ViG(num_classes, dim, depth, mlp_ratio, drop_path_rate, k=9)
    return model


def vig_s_patch16_224(args):
    """vig_s_patch16_224"""
    num_classes = args.num_classes
    dim = 320
    depth = 16
    mlp_ratio = 4
    drop_path_rate = args.drop_path_rate
    model = ViG(num_classes, dim, depth, mlp_ratio, drop_path_rate, k=9)
    return model


def vig_b_patch16_224(args):
    """vig_b_patch16_224"""
    num_classes = args.num_classes
    dim = 640
    depth = 16
    mlp_ratio = 4
    drop_path_rate = args.drop_path_rate
    model = ViG(num_classes, dim, depth, mlp_ratio, drop_path_rate, k=9)
    return model
