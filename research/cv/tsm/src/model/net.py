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
"""Net"""
import mindspore.nn as nn

from mindspore.common.initializer import Normal, Constant

import numpy as np
from src.utils.consensus import ConsensusModule
from src.utils.transforms import GroupRandomHorizontalFlip, GroupMultiScaleCrop
from src.utils.temporal_shift import make_temporal_shift
from src.utils.non_local import make_non_local
from src.model.resnet import resnet50


class TSM(nn.Cell):
    """
    Temporal Segment Network, base model of TSM.

    Args:
        in_planes (int): Input channel.
        ndf (int): Output channel.
        n_layers (int): The number of ConvNormReLU blocks.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".

    Returns:
        Tensor, output tensor.

    # Examples:
    #     >>> Discriminator(3, 64, 3)
    """

    def __init__(self, num_class, num_segments, modality, base_model='resnet50', new_length=None,
                 consensus_type='avg', before_softmax=True, dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, pretrain='imagenet', is_shift=False, shift_div=8,
                 shift_place='blockres', fc_lr5=False, temporal_pool=False, non_local=False, weight_init=True):
        super(TSM, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.weight_init = weight_init

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)

        self._prepare_tsm(num_class)


        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

        if self.is_shift and self.temporal_pool:
            self.new_segments = self.num_segments // 2
        else:
            self.new_segments = self.num_segments

    def _prepare_base_model(self, base_model):
        """_prepare_base_model"""
        if base_model == 'resnet50':
            self.base_model = resnet50()
            if self.is_shift:
                print('Adding temporal shift...')
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'end_point'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            # self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean([1, 2])]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        else:
            self.base_model = base_model
            if self.is_shift:
                print('Adding temporal shift...')
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'end_point'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            # self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean([1, 2])]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

    def _prepare_tsm(self, num_class):
        """_prepare_tsm"""
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_channels
        std = 0.001
        normal_initializer = Normal(sigma=std)
        constant_initializer = Constant(value=0)
        if self.weight_init:
            if self.dropout == 0:
                setattr(self.base_model, self.base_model.last_layer_name,
                        nn.Dense(feature_dim, num_class, weight_init=normal_initializer,
                                 bias_init=constant_initializer))
                self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=1 - self.dropout))
                self.new_fc = nn.Dense(feature_dim, num_class, weight_init=normal_initializer,
                                       bias_init=constant_initializer)
                # if hasattr(self.new_fc, 'weight'):
        else:
            if self.dropout == 0:
                setattr(self.base_model, self.base_model.last_layer_name,
                        nn.Dense(feature_dim, num_class, bias_init=constant_initializer))
                self.new_fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=1 - self.dropout))
                self.new_fc = nn.Dense(feature_dim, num_class, bias_init=constant_initializer)

        return feature_dim

    def set_train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSM, self).set_train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.cells():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.set_train(mode=False)
                        # shutdown update in frozen mode
                        m.gamma.requires_grad = False
                        m.beta.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        """get_optim_policies"""
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        conv_cnt = 0
        bn_cnt = 0

        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                ps = list(m.get_parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, nn.Dense):
                ps = list(m.get_parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.trainable_params()))

        return_dict = [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
        return [d for d in return_dict if d['params'] != []]

    def construct(self, inp):
        """construct"""
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length




        base_out = self.base_model(inp.view((-1, sample_len) + inp.shape[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        base_out = base_out.view((-1, self.new_segments) + base_out.shape[1:])
        output = self.consensus(base_out).squeeze(1)


        return output

    def _get_diff(self, inp, keep_rgb=False):
        """get_diff"""
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = inp.view((-1, self.num_segments, self.new_length + 1, input_c,) + inp.shape[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()
        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        return new_data


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        """get_augmentation"""
        if flip:
            return [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False)]
        print('#' * 20, 'NO FLIP!!!')
        return [GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])]
