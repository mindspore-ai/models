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

# This file was copied from project [ZhaoWeicheng][Pyramidbox.pytorch]

from mindspore import nn, ops, Parameter, Tensor
from mindspore.common import initializer
from mindspore import dtype as mstype

from src.loss import MultiBoxLoss

class L2Norm(nn.Cell):
    def __init__(self, n_channles, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channles
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = Parameter(Tensor(shape=(self.n_channels), init=initializer.Constant(value=self.gamma),
                                       dtype=mstype.float32))
        self.pow = ops.Pow()
        self.sum = ops.ReduceSum()
        self.div = ops.Div()

    def construct(self, x):
        norm = self.pow(x, 2).sum(axis=1, keepdims=True)
        norm = ops.sqrt(norm) + self.eps
        x = self.div(x, norm)
        out = self.weight[None, :][:, :, None][:, :, :, None].expand_as(x) * x
        return out

class ConvBn(nn.Cell):
    """docstring for conv"""

    def __init__(self,
                 in_plane,
                 out_plane,
                 kernel_size,
                 stride,
                 padding):
        super(ConvBn, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size, stride, pad_mode='pad',
                               padding=padding, has_bias=True, weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(out_plane)

    def construct(self, x):
        x = self.conv1(x)
        return self.bn1(x)

class CPM(nn.Cell):
    """docstring for CPM"""

    def __init__(self, in_plane):
        super(CPM, self).__init__()
        self.branch1 = ConvBn(in_plane, 1024, 1, 1, 0)
        self.branch2a = ConvBn(in_plane, 256, 1, 1, 0)
        self.branch2b = ConvBn(256, 256, 3, 1, 1)
        self.branch2c = ConvBn(256, 1024, 1, 1, 0)

        self.relu = nn.ReLU()

        self.ssh_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                               has_bias=True, weight_init='xavier_uniform')
        self.ssh_dimred = nn.Conv2d(1024, 128, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                    has_bias=True, weight_init='xavier_uniform')
        self.ssh_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                               has_bias=True, weight_init='xavier_uniform')
        self.ssh_3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True,
                                weight_init='xavier_uniform')
        self.ssh_3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                has_bias=True, weight_init='xavier_uniform')
        self.cat = ops.Concat(1)

    def construct(self, x):
        out_residual = self.branch1(x)
        x = self.relu(self.branch2a(x))
        x = self.relu(self.branch2b(x))
        x = self.branch2c(x)

        rescomb = self.relu(x + out_residual)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = self.relu(self.ssh_dimred(rescomb))
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = self.relu(self.ssh_3a(ssh_dimred))
        ssh_3b = self.ssh_3b(ssh_3a)

        ssh_out = self.cat((ssh1, ssh_2, ssh_3b))
        ssh_out = self.relu(ssh_out)
        return ssh_out


class PyramidBox(nn.Cell):
    """docstring for PyramidBox"""

    def __init__(self,
                 phase,
                 base,
                 extras,
                 lfpn_cpm,
                 head,
                 num_classes):
        super(PyramidBox, self).__init__()

        self.vgg = nn.CellList(base)
        self.extras = nn.CellList(extras)
        self.num_classes = num_classes

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.lfpn_topdown = nn.CellList(lfpn_cpm[0])
        self.lfpn_later = nn.CellList(lfpn_cpm[1])
        self.cpm = nn.CellList(lfpn_cpm[2])

        self.loc_layers = nn.CellList(head[0])
        self.conf_layers = nn.CellList(head[1])

        self.relu = nn.ReLU()
        self.concat = ops.Concat(1)

        self.is_infer = False

        if phase == 'test':
            self.softmax = nn.Softmax(axis=-1)
            self.is_infer = True

    def _upsample_prod(self, x, y):
        _, _, H, W = y.shape
        resize_bilinear = nn.ResizeBilinear()
        result = resize_bilinear(x, size=(H, W), align_corners=True) * y
        return result

    def construct(self, x):
        # apply vgg up to conv3_3 relu
        for k in range(16):
            x = self.vgg[k](x)
        conv3_3 = x
        # apply vgg up to conv4_3
        for k in range(16, 23):
            x = self.vgg[k](x)
        conv4_3 = x

        for k in range(23, 30):
            x = self.vgg[k](x)
        conv5_3 = x

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        convfc_7 = x
        # apply extra layers and cache source layer outputs
        for k in range(2):
            x = self.relu(self.extras[k](x))
        conv6_2 = x

        for k in range(2, 4):
            x = self.relu(self.extras[k](x))
        conv7_2 = x

        x = self.relu(self.lfpn_topdown[0](convfc_7))
        lfpn2_on_conv5 = self.relu(self._upsample_prod(
            x, self.lfpn_later[0](conv5_3)))

        x = self.relu(self.lfpn_topdown[1](lfpn2_on_conv5))
        lfpn1_on_conv4 = self.relu(self._upsample_prod(
            x, self.lfpn_later[1](conv4_3)))

        x = self.relu(self.lfpn_topdown[2](lfpn1_on_conv4))
        lfpn0_on_conv3 = self.relu(self._upsample_prod(
            x, self.lfpn_later[2](conv3_3)))


        ssh_conv3_norm = self.cpm[0](self.L2Norm3_3(lfpn0_on_conv3))
        ssh_conv4_norm = self.cpm[1](self.L2Norm4_3(lfpn1_on_conv4))
        ssh_conv5_norm = self.cpm[2](self.L2Norm5_3(lfpn2_on_conv5))
        ssh_convfc7 = self.cpm[3](convfc_7)
        ssh_conv6 = self.cpm[4](conv6_2)
        ssh_conv7 = self.cpm[5](conv7_2)

        face_locs, face_confs = [], []
        head_locs, head_confs = [], []

        N = ssh_conv3_norm.shape[0]
        mbox_loc = self.loc_layers[0](ssh_conv3_norm)
        face_loc, head_loc = ops.Split(axis=1, output_num=2)(mbox_loc)


        face_loc = ops.Transpose()(face_loc, (0, 2, 3, 1)).view(N, -1, 4)
        if not self.is_infer:
            head_loc = ops.Transpose()(head_loc, (0, 2, 3, 1)).view(N, -1, 4)

        mbox_conf = self.conf_layers[0](ssh_conv3_norm)
        face_conf1 = mbox_conf[:, 3:4, :, :]

        _, face_conf3_maxin = ops.ArgMaxWithValue(axis=1, keep_dims=True)(mbox_conf[:, 0:3, :, :])

        face_conf = self.concat((face_conf3_maxin, face_conf1))
        face_conf = ops.Transpose()(face_conf, (0, 2, 3, 1)).view(N, -1, 2)

        head_conf = None
        if not self.is_infer:
            _, head_conf3_maxin = ops.ArgMaxWithValue(axis=1, keep_dims=True)(mbox_conf[:, 4:7, :, :])
            head_conf1 = mbox_conf[:, 7:, :, :]
            head_conf = self.concat((head_conf3_maxin, head_conf1))
            head_conf = ops.Transpose()(head_conf, (0, 2, 3, 1)).view(N, -1, 2)

        face_locs.append(face_loc)
        face_confs.append(face_conf)

        if not self.is_infer:
            head_locs.append(head_loc)
            head_confs.append(head_conf)

        inputs = [ssh_conv4_norm, ssh_conv5_norm,
                  ssh_convfc7, ssh_conv6, ssh_conv7]

        feature_maps = []
        feat_size = ssh_conv3_norm.shape[2:]
        feature_maps.append([feat_size[0], feat_size[1]])

        for i, feat in enumerate(inputs):
            feat_size = feat.shape[2:]
            feature_maps.append([feat_size[0], feat_size[1]])
            mbox_loc = self.loc_layers[i + 1](feat)
            face_loc, head_loc = ops.Split(axis=1, output_num=2)(mbox_loc)
            face_loc = ops.Transpose()(face_loc, (0, 2, 3, 1)).view(N, -1, 4)
            if not self.is_infer:
                head_loc = ops.Transpose()(head_loc, (0, 2, 3, 1)).view(N, -1, 4)

            mbox_conf = self.conf_layers[i + 1](feat)
            face_conf1 = mbox_conf[:, 0:1, :, :]
            _, face_conf3_maxin = ops.ArgMaxWithValue(axis=1, keep_dims=True)(mbox_conf[:, 1:4, :, :])
            face_conf = self.concat((face_conf1, face_conf3_maxin))
            face_conf = ops.Transpose()(face_conf, (0, 2, 3, 1)).ravel().view(N, -1, 2)

            if not self.is_infer:
                head_conf = ops.Transpose()(mbox_conf[:, 4:, :, :], (0, 2, 3, 1)).view(N, -1, 2)

            face_locs.append(face_loc)
            face_confs.append(face_conf)

            if not self.is_infer:
                head_locs.append(head_loc)
                head_confs.append(head_conf)

        face_mbox_loc = self.concat(face_locs)
        face_mbox_conf = self.concat(face_confs)

        head_mbox_loc, head_mbox_conf = None, None
        if not self.is_infer:
            head_mbox_loc = self.concat(head_locs)
            head_mbox_conf = self.concat(head_confs)

        if not self.is_infer:
            output = (face_mbox_loc, face_mbox_conf, head_mbox_loc, head_mbox_conf)
        else:
            output = (face_mbox_loc, self.softmax(face_mbox_conf), feature_maps)
        return output

vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

lfpn_cpm_cfg = [256, 512, 512, 1024, 512, 256]

multibox_cfg = [512, 512, 512, 512, 512, 512]


def vgg_(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode='pad', padding=1,
                               has_bias=True, weight_init='xavier_uniform')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    conv6 = nn.Conv2d(512, 1024, kernel_size=3, pad_mode='pad', padding=6,
                      dilation=6, has_bias=True, weight_init='xavier_uniform')
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1, has_bias=True, weight_init='xavier_uniform')
    layers += [conv6, nn.ReLU(), conv7, nn.ReLU()]
    return layers


def add_extras(cfg, i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2,
                                     pad_mode='pad', padding=1, has_bias=True, weight_init='xavier_uniform')]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag],
                                     has_bias=True, weight_init='xavier_uniform')]
            flag = not flag
        in_channels = v
    return layers


def add_lfpn_cpm(cfg):
    lfpn_topdown_layers = []
    lfpn_latlayer = []
    cpm_layers = []

    for k, v in enumerate(cfg):
        cpm_layers.append(CPM(v))

    fpn_list = cfg[::-1][2:]
    for k, v in enumerate(fpn_list[:-1]):
        lfpn_latlayer.append(nn.Conv2d(fpn_list[k + 1], fpn_list[k + 1], kernel_size=1,
                                       stride=1, padding=0, has_bias=True, weight_init='xavier_uniform'))
        lfpn_topdown_layers.append(nn.Conv2d(v, fpn_list[k + 1], kernel_size=1, stride=1,
                                             padding=0, has_bias=True, weight_init='xavier_uniform'))

    return (lfpn_topdown_layers, lfpn_latlayer, cpm_layers)


def multibox(vgg, extra_layers):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, 28, -2]
    i = 0
    loc_layers += [nn.Conv2d(multibox_cfg[i], 8, kernel_size=3, pad_mode='pad', padding=1,
                             has_bias=True, weight_init='xavier_uniform')]
    conf_layers += [nn.Conv2d(multibox_cfg[i], 8, kernel_size=3, pad_mode='pad', padding=1,
                              has_bias=True, weight_init='xavier_uniform')]
    i += 1
    for _, _ in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(multibox_cfg[i], 8, kernel_size=3, pad_mode='pad', padding=1,
                                 has_bias=True, weight_init='xavier_uniform')]
        conf_layers += [nn.Conv2d(multibox_cfg[i], 6, kernel_size=3, pad_mode='pad', padding=1,
                                  has_bias=True, weight_init='xavier_uniform')]
        i += 1
    for _, _ in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(multibox_cfg[i], 8, kernel_size=3, pad_mode='pad',
                                 padding=1, has_bias=True, weight_init='xavier_uniform')]
        conf_layers += [nn.Conv2d(multibox_cfg[i], 6, kernel_size=3, pad_mode='pad', padding=1,
                                  has_bias=True, weight_init='xavier_uniform')]
        i += 1
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_net(phase, num_classes=2):
    base_, extras_, head_ = multibox(vgg_(vgg_cfg, 3), add_extras((extras_cfg), 1024))
    lfpn_cpm = add_lfpn_cpm(lfpn_cpm_cfg)
    return PyramidBox(phase, base_, extras_, lfpn_cpm, head_, num_classes)

class NetWithLoss(nn.Cell):
    def __init__(self, net):
        super(NetWithLoss, self).__init__()
        self.net = net
        self.loss_fn_1 = MultiBoxLoss()
        self.loss_fn_2 = MultiBoxLoss(use_head_loss=True)

    def construct(self, images, face_loc, face_conf, head_loc, head_conf):
        out = self.net(images)
        face_loss_l, face_loss_c = self.loss_fn_1(out, (face_loc, face_conf))
        head_loss_l, head_loss_c = self.loss_fn_2(out, (head_loc, head_conf))
        loss = face_loss_l + face_loss_c + head_loss_l + head_loss_c
        return loss

class EvalLoss(nn.Cell):
    """
    Calculate loss value while training.
    """
    def __init__(self, net):
        super(EvalLoss, self).__init__()
        self.net = net
        self.loss_fn = MultiBoxLoss()

    def construct(self, images, face_loc, face_conf):
        out = self.net(images)
        face_loss_l, face_loss_c = self.loss_fn(out, (face_loc, face_conf))
        loss = face_loss_l + face_loss_c
        return loss
