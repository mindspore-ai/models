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
"""Network modules and utilities"""

from mindspore import load_checkpoint
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer


class VGG(nn.Cell):

    def __init__(self, cfg, i, batch_norm=False, pretrained=None):
        super().__init__()
        self.layers = self.make_layers(cfg, i, batch_norm=batch_norm)
        if pretrained:
            print('Loading pretrained VGG16...')
            load_checkpoint(pretrained, self)

    def make_layers(self, cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='same')
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, pad_mode='pad', padding=6, dilation=6, has_bias=True)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1, has_bias=True)
        layers += [pool5, conv6,
                   nn.ReLU(), conv7, nn.ReLU()]
        return nn.CellList(layers)

    def construct(self, x, out_inds):
        out = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in out_inds:
                out.append(x)
        return out


class BasicConv(nn.Cell):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, group=groups, has_bias=bias,
                              pad_mode='pad')
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.99, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class TUM(nn.Cell):

    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super().__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes
        self.concat = ops.Concat(axis=1)

        layers = [BasicConv(self.in1, self.planes, 3, 2, 1)]
        for i in range(self.scales - 2):
            if not i == self.scales - 3:
                layers.append(BasicConv(self.planes, self.planes, 3, 2, 1))
            else:
                layers.append(BasicConv(self.planes, self.planes, 3, 1, 0))
        self.layers = nn.CellList(layers)
        self.n_layers = len(layers)
        self.toplayer = nn.CellList([BasicConv(self.planes, self.planes, 1, 1, 0)])

        latlayer = []
        for i in range(self.scales - 2):
            latlayer.append(BasicConv(self.planes, self.planes, 3, 1, 1))
        latlayer.append(BasicConv(self.in1, self.planes, 3, 1, 1))
        self.latlayer = nn.CellList(latlayer)

        if self.is_smooth:
            smooth = []
            for i in range(self.scales - 1):
                smooth.append(BasicConv(self.planes, self.planes, 1, 1, 0))
            self.smooth = nn.CellList(smooth)

    def _upsample_add(self, x, y):
        H, W = y.shape[-2:]
        out = ops.ResizeNearestNeighbor((H, W))(x) + y
        return out

    def construct(self, x, y):
        if not self.first_level:
            x = self.concat([x, y])
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)

        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                self._upsample_add(
                    deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers) - 1 - i])
                    )
            )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                    self.smooth[i](deconved_feat[i + 1])
                    )
            return smoothed_feat
        return deconved_feat


class DynamicUpscale(nn.Cell):

    def __init__(self, scale_factor=1, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def construct(self, x):
        shape = x.shape[-2:]
        if self.mode == 'nearest':
            operation = ops.ResizeNearestNeighbor((shape[0]*self.scale_factor, shape[1]*self.scale_factor))(x)
        else:
            operation = nn.ResizeBilinear()(x, size=(shape[0]*self.scale_factor, shape[1]*self.scale_factor))
        return operation


class M2Det(nn.Cell):

    def __init__(self, phase, size, config=None):
        """M2Det: Multi-level Multi-scale single-shot object Detector"""
        super().__init__()
        self.model_phase = phase
        self.size = size
        self.init_params(config)
        self.construct_modules()
        self.concat = ops.Concat(axis=1)
        self.upscale = DynamicUpscale(scale_factor=2, mode='nearest')

    def init_params(self, config=None):  # Directly read the config
        assert config is not None, 'Error: no config'
        for key, value in config.items():
            setattr(self, key, value)

    def construct_modules(self):
        # construct tums
        for i in range(self.num_levels):
            if i == 0:
                setattr(self,
                        'unet{}'.format(i + 1),
                        TUM(first_level=True,
                            input_planes=self.planes // 2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=512))  # side channel isn't fixed.
            else:
                setattr(self,
                        'unet{}'.format(i + 1),
                        TUM(first_level=False,
                            input_planes=self.planes // 2,
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=self.planes))
        self.unets = []
        for i in range(self.num_levels):
            self.unets.append(getattr(self, 'unet{}'.format(i + 1)))

        # construct base features
        if 'vgg' in self.net_family:
            if self.backbone == 'vgg16':
                vgg_param = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
                self.base = VGG(vgg_param, 3, batch_norm=False, pretrained=self.checkpoint_path)
                shallow_in, shallow_out = 512, 256
                deep_in, deep_out = 1024, 512
            else:
                print(f'Backbone {self.backbone} not implemented')
        else:
            print(f'Net family {self.net_family} not implemented')
        self.reduce = BasicConv(shallow_in, shallow_out, 3, stride=1, padding=1)
        self.up_reduce = BasicConv(deep_in, deep_out, 1, stride=1)

        # construct others
        if self.model_phase == 'test':
            self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256 * 8)
        self.leach = nn.CellList([BasicConv(
            deep_out + shallow_out,
            self.planes // 2,
            kernel_size=(1, 1), stride=(1, 1))] * self.num_levels)

        # construct localization and recognition layers
        loc_ = list()
        conf_ = list()
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(in_channels=self.planes * self.num_levels,
                                  out_channels=4 * 6,  # 4 is coordinates, 6 is anchors for each pixels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  pad_mode='pad',
                                  has_bias=True,
                                  weight_init='uniform'))
            conf_.append(nn.Conv2d(in_channels=self.planes * self.num_levels,
                                   out_channels=self.num_classes * 6,  # 6 is anchors for each pixels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   pad_mode='pad',
                                   has_bias=True,
                                   weight_init='uniform'))
        self.loc = nn.CellList(loc_)
        self.conf = nn.CellList(conf_)

    def construct(self, x):
        loc, conf, base_feats = [], [], []
        base_feats = self.base(x, self.base_out)
        base_feature = self.concat(
            (self.reduce(base_feats[0]), self.upscale(self.up_reduce(base_feats[1]))))

        # tum_outs is the multi-level multi-scale feature
        tum_outs = [self.unets[0](self.leach[0](base_feature), 'none')]
        for i in range(1, self.num_levels, 1):
            tum_outs.append(
                self.unets[i](
                    self.leach[i](base_feature), tum_outs[i - 1][-1]
                    )
            )
        # concat with same scales
        sources = []
        for i in range(self.num_scales, 0, -1):
            _fx_list = []
            for j in range(self.num_levels):
                _fx_list.append(tum_outs[j][i - 1])
            sources.append(self.concat(_fx_list))

        sources[0] = self.Norm(sources[0])

        for (k, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(k).transpose(0, 2, 3, 1))
            conf.append(c(k).transpose(0, 2, 3, 1))

        loc_list = []
        conf_list = []
        for i in range(self.num_scales):
            loc_list.append(loc[i].view(loc[i].shape[0], -1))
            conf_list.append(conf[i].view(conf[i].shape[0], -1))
        loc = self.concat(loc_list)
        conf = self.concat(conf_list)

        if self.model_phase == "test":
            output = (
                loc.view(loc.shape[0], -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.shape[0], -1, 4),
                conf.view(conf.shape[0], -1, self.num_classes),
            )
        return output

    def init_model(self):
        def weights_init(m):
            for _, cell in m.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=0.001),
                                                                 cell.weight.shape,
                                                                 cell.weight.dtype))
                    if cell.has_bias:
                        cell.bias.set_data(initializer.initializer(0,
                                                                   cell.bias.shape,
                                                                   cell.bias.dtype))
                elif isinstance(cell, nn.BatchNorm2d):
                    cell.gamma.set_data(initializer.initializer(1,
                                                                cell.gamma.shape,
                                                                cell.gamma.dtype))
                    cell.beta.set_data(initializer.initializer(0,
                                                               cell.beta.shape,
                                                               cell.beta.dtype))

        print('Initializing weights for [tums, reduce, up_reduce, leach, loc, conf]...')
        for i in range(self.num_levels):
            weights_init(self.unets[i])
        weights_init(self.reduce)
        weights_init(self.up_reduce)
        weights_init(self.leach)
        weights_init(self.loc)
        weights_init(self.conf)


class M2DetWithLoss(nn.Cell):

    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def construct(self, img, loc, conf):
        output = self.model(img)
        return self.loss(output, loc, conf)


def get_model(cfg, input_size, test=False):
    if test:
        phase = 'test'
    else:
        phase = 'train'
    model = M2Det(phase, input_size, config=cfg)
    model.init_model()

    return model
