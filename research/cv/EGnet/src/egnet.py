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

"""EGNet model define"""

import mindspore
import mindspore.common.initializer as init
from mindspore import nn, load_checkpoint

from src.resnet import resnet50
from src.vgg import Vgg16
import numpy as np

config_vgg = {"convert": [[128, 256, 512, 512, 512], [64, 128, 256, 512, 512]],
              "merge1": [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],
                         [512, 0, 512, 7, 3]], "merge2": [[128], [256, 512, 512, 512]]}  # no convert layer, no conv6

config_resnet = {"convert": [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]],
                 "deep_pool": [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False],
                               [True, True, True, True, False]], "score": 256,
                 "edgeinfo": [[16, 16, 16, 16], 128, [16, 8, 4, 2]], "edgeinfoc": [64, 128],
                 "block": [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], "fuse": [[16, 16, 16, 16], True],
                 "fuse_ratio": [[16, 1], [8, 1], [4, 1], [2, 1]],
                 "merge1": [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],
                            [512, 0, 512, 7, 3]], "merge2": [[128], [256, 512, 512, 512]]}


class ConvertLayer(nn.Cell):
    """
    Convert layer
    """
    def __init__(self, list_k):
        """
        initialize convert layer for resnet config
        """
        super(ConvertLayer, self).__init__()
        up0 = []
        for i in range(len(list_k[0])):
            up0.append(nn.SequentialCell([nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, has_bias=False), nn.ReLU()]))
        self.convert0 = nn.CellList(up0)

    def construct(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        tuple_resl = ()
        for i in resl:
            tuple_resl += (i,)
        return tuple_resl


class MergeLayer1(nn.Cell):
    """
    merge layer 1
    """
    def __init__(self, list_k):
        """
        initialize merge layer 1
        @param list_k: [[64, 512, 64], [128, 512, 128], [256, 0, 256] ... ]
        """
        super(MergeLayer1, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for ik in list_k:
            if ik[1] > 0:
                trans.append(nn.SequentialCell([nn.Conv2d(ik[1], ik[0], 1, 1, has_bias=False), nn.ReLU()]))
            up.append(nn.SequentialCell(
                [nn.Conv2d(ik[0], ik[2], ik[3], 1, has_bias=True, pad_mode="pad", padding=ik[4]), nn.ReLU(),
                 nn.Conv2d(ik[2], ik[2], ik[3], 1, has_bias=True, pad_mode="pad", padding=ik[4]), nn.ReLU(),
                 nn.Conv2d(ik[2], ik[2], ik[3], 1, has_bias=True, pad_mode="pad", padding=ik[4]), nn.ReLU()]))
            score.append(nn.Conv2d(ik[2], 1, 3, 1, pad_mode="pad", padding=1, has_bias=True))
        trans.append(nn.SequentialCell([nn.Conv2d(512, 128, 1, 1, has_bias=False), nn.ReLU()]))
        self.trans, self.up, self.score = nn.CellList(trans), nn.CellList(up), nn.CellList(score)
        self.relu = nn.ReLU()
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, list_x, x_size):
        """
        forward
        """
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []

        num_f = len(list_x)
        # Conv6-3 Conv
        tmp = self.up[num_f - 1](list_x[num_f - 1])
        sal_feature.append(tmp)
        u_tmp = tmp

        # layer6 -> layer0
        up_sal.append(self.resize_bilinear(self.score[num_f - 1](tmp), x_size, align_corners=True))

        # layer5 layer4 layer3
        for j in range(2, num_f):
            i = num_f - j
            # different channel, use trans layer, or resize and add directly
            if list_x[i].shape[1] < u_tmp.shape[1]:
                u_tmp = list_x[i] + self.resize_bilinear((self.trans[i](u_tmp)), list_x[i].shape[2:],
                                                         align_corners=True)
            else:
                u_tmp = list_x[i] + self.resize_bilinear(u_tmp, list_x[i].shape[2:], align_corners=True)
            # Conv
            tmp = self.up[i](u_tmp)
            u_tmp = tmp
            sal_feature.append(tmp)
            up_sal.append(self.resize_bilinear(self.score[i](tmp), x_size, align_corners=True))

        u_tmp = list_x[0] + self.resize_bilinear(self.trans[-1](sal_feature[0]), list_x[0].shape[2:],
                                                 align_corners=True)
        tmp = self.up[0](u_tmp)
        # layer 2
        edge_feature.append(tmp)
        up_edge.append(self.resize_bilinear(self.score[0](tmp), x_size, align_corners=True))
        tuple_up_edge, tuple_edge_feature, tuple_up_sal, tuple_sal_feature = (), (), (), ()
        for i in up_edge:
            tuple_up_edge += (i,)
        for i in edge_feature:
            tuple_edge_feature += (i,)
        for i in up_sal:
            tuple_up_sal += (i,)
        for i in sal_feature:
            tuple_sal_feature += (i,)

        return tuple_up_edge, tuple_edge_feature, tuple_up_sal, tuple_sal_feature


class MergeLayer2(nn.Cell):
    """
    merge layer 2
    """
    def __init__(self, list_k):
        """
        initialize merge layer 2
        """
        super(MergeLayer2, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3, 1], [5, 2], [5, 2], [7, 3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.SequentialCell([nn.Conv2d(j, i, 1, 1, has_bias=False), nn.ReLU()]))

                tmp_up.append(
                    nn.SequentialCell([nn.Conv2d(i, i, feature_k[idx][0], 1, pad_mode="pad", padding=feature_k[idx][1],
                                                 has_bias=True), nn.ReLU(),
                                       nn.Conv2d(i, i, feature_k[idx][0], 1, pad_mode="pad", padding=feature_k[idx][1],
                                                 has_bias=True), nn.ReLU(),
                                       nn.Conv2d(i, i, feature_k[idx][0], 1, pad_mode="pad", padding=feature_k[idx][1],
                                                 has_bias=True), nn.ReLU()]))
                tmp_score.append(nn.Conv2d(i, 1, 3, 1, pad_mode="pad", padding=1, has_bias=True))
            trans.append(nn.CellList(tmp))
            up.append(nn.CellList(tmp_up))
            score.append(nn.CellList(tmp_score))

        self.trans, self.up, self.score = nn.CellList(trans), nn.CellList(up), nn.CellList(score)

        self.final_score = nn.SequentialCell([nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, has_bias=True), nn.ReLU(),
                                              nn.Conv2d(list_k[0][0], 1, 3, 1, has_bias=True)])
        self.relu = nn.ReLU()
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, list_x, list_y, x_size):
        """
        forward
        """
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):
                tmp = self.resize_bilinear(self.trans[i][j](j_x), i_x.shape[2:], align_corners=True) + i_x
                tmp_f = self.up[i][j](tmp)
                up_score.append(self.resize_bilinear(self.score[i][j](tmp_f), x_size, align_corners=True))
                tmp_feature.append(tmp_f)

        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(tmp_fea + self.resize_bilinear(tmp_feature[i_fea + 1], tmp_feature[0].shape[2:],
                                                               align_corners=True))
        up_score.append(self.resize_bilinear(self.final_score(tmp_fea), x_size, align_corners=True))
        return up_score


class EGNet(nn.Cell):
    """
    EGNet network
    """
    def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
        """ initialize
        """
        super(EGNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        if self.base_model_cfg == "resnet":
            self.convert = ConvertLayer(config_resnet["convert"])
        self.base = base
        self.merge1 = merge1_layers
        self.merge2 = merge2_layers

    def construct(self, x):
        """
        forward
        """
        x_size = x.shape[2:]
        conv2merge = self.base(x)
        if self.base_model_cfg == "resnet":
            conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        tuple_up_edge, tuple_up_sal, tuple_up_sal_final = (), (), ()
        for i in up_edge:
            tuple_up_edge += (i,)
        for i in up_sal:
            tuple_up_sal += (i,)
        for i in up_sal_final:
            tuple_up_sal_final += (i,)

        return tuple_up_edge, tuple_up_sal, tuple_up_sal_final

    def load_pretrained_model(self, model_file):
        """
        load pretrained model
        """
        load_checkpoint(model_file, net=self)


def extra_layer(base_model_cfg, base):
    """
    extra layer for different base network
    """
    if base_model_cfg == "vgg":
        config = config_vgg
    elif base_model_cfg == "resnet":
        config = config_resnet
    else:
        raise ValueError(f"{base_model_cfg} backbone is not implemented")
    merge1_layers = MergeLayer1(config["merge1"])
    merge2_layers = MergeLayer2(config["merge2"])

    return base, merge1_layers, merge2_layers


def build_model(base_model_cfg="vgg"):
    """
    build the whole network
    """
    if base_model_cfg == "vgg":
        return EGNet(base_model_cfg, *extra_layer(base_model_cfg, Vgg16()))
    if base_model_cfg == "resnet":
        return EGNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50()))
    raise ValueError("unknown config")


def init_weights(net, init_type="normal", init_gain=0.01, constant=0.001):
    """
    Initialize network weights.
    """
    np.random.seed(1)
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            if init_type == "normal":
                cell.weight.set_data(
                    mindspore.Tensor(np.random.normal(0, 0.01, size=cell.weight.shape), dtype=cell.weight.dtype))
            elif init_type == "xavier":
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(init_gain), cell.weight.shape, cell.weight.dtype))
            elif init_type == "constant":
                cell.weight.set_data(init.initializer(constant, cell.weight.shape, cell.weight.dtype))
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.gamma.dtype))


if __name__ == "__main__":
    model = build_model()
    print(model)
