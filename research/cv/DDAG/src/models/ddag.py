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
"""model_main.py"""
import mindspore as ms
import mindspore.common.initializer as init
import mindspore.ops as P

from mindspore import nn
from mindspore.common.initializer import Normal

from src.models.resnet import resnet50, resnet50_share, resnet50_specific
from src.models.attention import IWPA, GraphAttentionLayer


def weights_init_kaiming(m):
    """
    function of weights_init_kaiming
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_data(init.initializer(init.HeNormal(
            negative_slope=0, mode='fan_in'), m.weight.shape, m.weight.dtype))
    elif classname.find('Linear') != -1:
        m.weight.set_data(init.initializer(init.HeNormal(
            negative_slope=0, mode='fan_out'), m.weight.shape, m.weight.dtype))
        m.bias.set_data(init.initializer(
            init.Zero(), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        m.gamma.set_data(init.initializer(Normal(
            mean=1.0, sigma=0.01), m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(init.initializer(
            init.Zero(), m.beta.shape, m.beta.dtype))


def weights_init_classifier(m):
    """
    function of weights_init_classifier
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.gamma.set_data(init.initializer(init.Normal(
            sigma=0.001), m.gamma.shape, m.gamma.dtype))
        if m.bias:
            m.bias.set_data(init.initializer(
                init.Zero(), m.bias.shape, m.bias.dtype))


class Normalize(nn.Cell):
    """
    class of normalize
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        self.pow = P.Pow()
        self.sum = P.ReduceSum(keep_dims=True)
        self.div = P.Div()

    def construct(self, x):
        norm = self.pow(x, self.power)
        norm = self.sum(norm, 1)
        norm = self.pow(norm, 1. / self.power)
        out = self.div(x, norm)
        return out


class Visible(nn.Cell):
    """
    class of visible module
    """
    def __init__(self, pretrain=""):
        super(Visible, self).__init__()
        self.visible = resnet50_specific(pretrain=pretrain)

    def construct(self, x):

        x = self.visible(x)
        return x


class Thermal(nn.Cell):
    """
    class of thermal_module
    """
    def __init__(self, pretrain=""):
        super(Thermal, self).__init__()

        self.thermal = resnet50_specific(pretrain=pretrain)

    def construct(self, x):

        x = self.thermal(x)

        return x


class BASE(nn.Cell):
    def __init__(self, pretrain=""):
        super(BASE, self).__init__()
        self.base = resnet50_share(pretrain=pretrain)

    def construct(self, x):
        x = self.base(x)
        return x


class ResNet50(nn.Cell):
    def __init__(self, pretrain=""):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrain=pretrain)

    def construct(self, x):
        x = self.resnet(x)
        return x


class DDAG(nn.Cell):
    """
    class of DDAG
    """
    def __init__(self, low_dim, class_num=200, drop=0.2, part=0, alpha=0.2, nheads=4, pretrain=""):
        super(DDAG, self).__init__()
        self.thermal_module = Thermal(pretrain=pretrain)
        self.visible_module = Visible(pretrain=pretrain)
        self.base_resnet = BASE(pretrain=pretrain)
        self.resnet50 = resnet50(pretrain=pretrain)
        pool_dim = 2048
        self.dropout = drop
        self.part = part
        self.class_num = class_num
        self.nheads = nheads

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(num_features=pool_dim)
        self.bottleneck.requires_grad = False
        self.classifier = nn.Dense(pool_dim, class_num, has_bias=False)

        weights_init_kaiming(self.bottleneck)
        weights_init_classifier(self.classifier)

        self.avgpool = P.ReduceMean(keep_dims=True)
        if self.part > 0:
            self.wpa = IWPA(pool_dim, self.part)
        else:
            self.wpa = IWPA(pool_dim, 3)

        self.cat = P.Concat()
        self.logsoftmax = nn.LogSoftmax()

        if nheads > 0:
            self.graph_att = GraphAttentionLayer(
                class_num, nheads, pool_dim, low_dim, drop, alpha)
        else:
            self.graph_att = GraphAttentionLayer(
                class_num, 4, pool_dim, low_dim, drop, alpha)

    # @profile
    def construct(self, x1, x2=None, adj=None, modal=0):
        """
        function of constructing
        """
        x = None
        feat_att = None
        out_att = None
        out_graph = None

        # modify version
        if modal == 0:
            x = self.cat((x1, x2))
        elif modal == 1:
            x = x1
        else:
            x = x2
        x = self.resnet50(x)
        x_pool = self.avgpool(x, (2, 3))
        x_pool = x_pool.view(x_pool.shape[0], x_pool.shape[1])
        feat = self.bottleneck(x_pool)  # mindspore version >=1.3.0

        if self.part > 0:
            # intra_modality weighted part attention
            feat_att = self.wpa(x, feat, 1)

        if self.training:
            if self.nheads > 0:
                # cross-modality graph attention
                out_graph = self.logsoftmax(self.graph_att(feat, adj))

            out = self.classifier(feat)
            # print("resnet classification output is", out)
            if self.part > 0:
                out_att = self.classifier(feat_att)
                # print("IWPA classification output is", out_att)

            if (self.part > 0) and (self.nheads > 0):
                return feat, feat_att, out, out_att, out_graph

            if self.nheads > 0:
                return feat, feat, out, out, out_graph

            if self.part > 0:
                return feat, feat_att, out, out_att
            return feat, feat, out, out  # just for debug

        # inference
        if self.part > 0:
            return self.l2norm(feat), self.l2norm(feat_att)
        return self.l2norm(feat), self.l2norm(feat)  # just for debug

    def create_graph(self, target1, target2):
        """
        Graph Construction
        """
        target = P.Cast()(self.cat([target1, target2]), ms.int32)
        one_hot = P.Gather()(P.Eye()(self.class_num, self.class_num, ms.float32), target, 0)
        one_hot_tran = one_hot.transpose()
        adj = ms.ops.matmul(one_hot, one_hot_tran) + \
            P.Eye()(target.shape[0], target.shape[0], ms.float32)
        # adjacent matrix normalize
        norm = P.Pow()(P.Pow()(adj, 2).sum(axis=1, keepdims=True), 1. / 2)
        adj_norm = P.Div()(adj, norm)
        return adj_norm
