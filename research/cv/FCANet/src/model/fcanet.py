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
fcanet network
"""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import ReduceMean
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore import Parameter
from src.model.res2net import res2net101

ResizeFunc = P.ResizeBilinear


#######################################[ FCANet ]#######################################


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, weight_init="he_normal"
    )


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """3x3 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        pad_mode="same",
        padding=0,
        dilation=dilation,
        weight_init="he_normal",
    )


class ASPPConv(nn.Cell):
    """ASPP convolution"""

    def __init__(
            self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True
    ):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                has_bias=False,
                weight_init="he_normal",
            )
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                pad_mode="pad",
                padding=atrous_rate,
                dilation=atrous_rate,
                weight_init="he_normal",
            )

        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def construct(self, x):
        out = self.aspp_conv(x)
        return out


class ASPPPooling(nn.Cell):
    """ASPP pooling"""

    def __init__(self, in_channels, out_channels, use_batch_statistics=True):
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, weight_init="he_normal"
                ),
                nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
                nn.ReLU(),
            ]
        )
        self.shape = P.Shape()

    def construct(self, x):
        out = nn.AvgPool2d(x.shape[2:])(x)
        out = self.conv(out)
        out = ResizeFunc(x.shape[2:], True)(out)
        return out


class ASPP(nn.Cell):
    """ASPP module"""

    def __init__(
            self,
            atrous_rates,
            in_channels=2048,
            out_channels=256,
            use_batch_statistics=True,
    ):
        super(ASPP, self).__init__()
        self.aspp1 = ASPPConv(
            in_channels,
            out_channels,
            atrous_rates[0],
            use_batch_statistics=use_batch_statistics,
        )
        self.aspp2 = ASPPConv(
            in_channels,
            out_channels,
            atrous_rates[1],
            use_batch_statistics=use_batch_statistics,
        )
        self.aspp3 = ASPPConv(
            in_channels,
            out_channels,
            atrous_rates[2],
            use_batch_statistics=use_batch_statistics,
        )
        self.aspp4 = ASPPConv(
            in_channels,
            out_channels,
            atrous_rates[3],
            use_batch_statistics=use_batch_statistics,
        )
        self.aspp_pooling = ASPPPooling(in_channels, out_channels)
        self.conv1 = nn.Conv2d(
            out_channels * (len(atrous_rates) + 1),
            out_channels,
            kernel_size=1,
            weight_init="he_normal",
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels, use_batch_statistics=use_batch_statistics
        )
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """ASPP construct"""
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)
        x = self.concat((x1, x2))
        x = self.concat((x, x3))
        x = self.concat((x, x4))
        x = self.concat((x, x5))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    """decoder module"""

    def __init__(self, in_ch, side_ch, side_ch_reduce, out_ch, use_batch_statistics):
        super(Decoder, self).__init__()
        self.side_conv = conv1x1(side_ch, side_ch_reduce)
        self.side_bn = nn.BatchNorm2d(
            side_ch_reduce, use_batch_statistics=use_batch_statistics
        )
        self.merge_conv1 = conv3x3(in_ch + side_ch_reduce, out_ch)
        self.merge_bn1 = nn.BatchNorm2d(
            out_ch, use_batch_statistics=use_batch_statistics
        )
        self.merge_conv2 = conv3x3(out_ch, out_ch)
        self.merge_bn2 = nn.BatchNorm2d(
            out_ch, use_batch_statistics=use_batch_statistics
        )
        self.relu = nn.ReLU()
        self.shape = P.Shape()
        self.concat = P.Concat(axis=1)

    def construct(self, x, side):
        """Decoder construct"""
        side = self.side_conv(side)
        side = self.side_bn(side)
        side = self.relu(side)
        x = ResizeFunc(side.shape[2:], True)(x)
        x = self.concat((x, side))
        x = self.merge_conv1(x)
        x = self.merge_bn1(x)
        x = self.relu(x)
        x = self.merge_conv2(x)
        x = self.merge_bn2(x)
        x = self.relu(x)
        return x


class PredDecoder(nn.Cell):
    """predict module"""

    def __init__(self, in_ch, use_batch_statistics):
        super(PredDecoder, self).__init__()
        self.conv1 = conv3x3(in_ch, in_ch // 2)
        self.bn1 = nn.BatchNorm2d(in_ch // 2, use_batch_statistics=use_batch_statistics)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.bn2 = nn.BatchNorm2d(in_ch // 2, use_batch_statistics=use_batch_statistics)
        self.conv3 = conv1x1(in_ch // 2, 1)
        self.relu = nn.ReLU()

    def construct(self, x):
        """predict construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class FcaModule(nn.Cell):
    """first click attention module"""

    def __init__(self, in_ch, use_batch_statistics):
        super(FcaModule, self).__init__()
        self.conv1 = conv3x3(in_ch, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics)
        self.conv2 = conv3x3(256, 256)
        self.bn2 = nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics)
        self.conv3 = conv3x3(256, 256)
        self.bn3 = nn.BatchNorm2d(256, use_batch_statistics=use_batch_statistics)
        self.conv4 = conv3x3(256, 512, stride=2)
        self.bn4 = nn.BatchNorm2d(512, use_batch_statistics=use_batch_statistics)
        self.conv5 = conv3x3(512, 512)
        self.bn5 = nn.BatchNorm2d(512, use_batch_statistics=use_batch_statistics)
        self.conv6 = conv3x3(512, 512)
        self.bn6 = nn.BatchNorm2d(512, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()

    def construct(self, x):
        """first click attention module construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        return x


def get_mask_gauss(mask_dist_src, sigma):
    """generate gauss mask from distance mask"""
    return P.Exp()(-2.772588722 * (mask_dist_src ** 2) / (sigma ** 2))


class FCANet(nn.Cell):
    """ main network"""

    def __init__(self, size=512, backbone_pretrained=None):
        super(FCANet, self).__init__()
        use_batch_statistics = None
        resnet = res2net101(input_channels=5)
        if backbone_pretrained is not None:
            resnet.load_pretrained_model(backbone_pretrained)
        self.resnet = resnet

        self.aspp = ASPP(
            [max(int(i * size / 512 + 0.5), 1) for i in [1, 6, 12, 18]],
            2048 + 512,
            256,
            use_batch_statistics=use_batch_statistics,
        )
        self.decoder = Decoder(
            in_ch=256,
            side_ch=256,
            side_ch_reduce=48,
            out_ch=256,
            use_batch_statistics=use_batch_statistics,
        )
        self.pred_decoder = PredDecoder(
            in_ch=256, use_batch_statistics=use_batch_statistics
        )
        self.first_conv = FcaModule(256 + 1, use_batch_statistics=use_batch_statistics)
        self.first_pred_decoder = PredDecoder(
            in_ch=512, use_batch_statistics=use_batch_statistics
        )
        self.concat = P.Concat(axis=1)
        self.shape = P.Shape()

    def construct(self, img, pos_mask_dist_src, neg_mask_dist_src, pos_mask_dist_first):
        """ main network construct"""
        img_with_anno = self.concat((img, get_mask_gauss(pos_mask_dist_src, 10)))
        img_with_anno = self.concat(
            (img_with_anno, get_mask_gauss(neg_mask_dist_src, 10))
        )
        l1, _, _, l4 = self.resnet(img_with_anno)

        first_map = ResizeFunc(l1.shape[2:], True)(
            get_mask_gauss(pos_mask_dist_first, 30)
        )
        l1_first = self.concat((l1, first_map))
        l1_first = self.first_conv(l1_first)
        result_first = self.first_pred_decoder(l1_first)
        result_first = ResizeFunc(img.shape[2:], True)(result_first)

        l4 = self.concat((l1_first, l4))
        x = self.aspp(l4)
        x = self.decoder(x, l1)
        x = self.pred_decoder(x)
        x = ResizeFunc(img.shape[2:], True)(x)
        return [x, result_first]


#######################################[ FCANet Training Module]#######################################


def _get_parallel_mode():
    """Get parallel mode."""
    return auto_parallel_context().get_parallel_mode()


class MyWithLossCell(nn.Cell):
    """ network with loss"""

    def __init__(self, backbone, loss_fn, batch_size=8, size=384):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.minimum = P.Minimum()
        self.maximum = P.Maximum()
        self.assign = P.Assign()
        self.out_tmp = Parameter(
            np.ones([batch_size, 1, size, size], dtype=np.float32),
            name="out",
            requires_grad=False,
        )

    def construct(
            self,
            img,
            pos_mask_dist_src,
            neg_mask_dist_src,
            pos_mask_dist_first,
            gt,
            click_loss_weight,
            first_loss_weight,
    ):
        """ MyWithLossCell construct"""
        out = self._backbone(
            img, pos_mask_dist_src, neg_mask_dist_src, pos_mask_dist_first
        )
        loss = ReduceMean(False)(
            self._loss_fn(out[0], gt) * click_loss_weight
        ) + ReduceMean(False)(self._loss_fn(out[1], gt) * first_loss_weight)
        return F.depend(loss, self.assign(self.out_tmp, out[0]))


class MyTrainOneStepCell(nn.Cell):
    """ cell for training one step """

    def __init__(self, network_with_loss, network, criterion, optimizer, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network_with_loss = network_with_loss
        self.network = network
        self.criterion = criterion
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_mirror_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(
                optimizer.parameters, mean, degree
            )
        self.minimum = P.Minimum()
        self.maximum = P.Maximum()

    def construct(
            self,
            img,
            pos_mask_dist_src,
            neg_mask_dist_src,
            pos_mask_dist_first,
            gt,
            click_loss_weight,
            first_loss_weight,
    ):
        """ MyTrainOneStepCell construct"""
        weights = self.weights
        loss = self.network_with_loss(
            img,
            pos_mask_dist_src,
            neg_mask_dist_src,
            pos_mask_dist_first,
            gt,
            click_loss_weight,
            first_loss_weight,
        )
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network_with_loss, weights)(
            img,
            pos_mask_dist_src,
            neg_mask_dist_src,
            pos_mask_dist_first,
            gt,
            click_loss_weight,
            first_loss_weight,
            sens,
        )
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
