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
"""
Get model object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mindspore

from src.i3d import InceptionI3D
from src.i3d import get_fine_tuning_parameters


def get_model(config):
    assert config.mode in ['rgb', 'flow']
    print('Mode:{}  Initializing i3d model (num_classes={})...'.format(config.mode, config.num_classes))

    if config.mode == 'rgb':
        in_channels = 3
    else:
        in_channels = 2

    model = InceptionI3D(
        is_train=True,
        amp_level=config.amp_level,
        num_classes=config.num_classes,
        train_spatial_squeeze=True,
        final_endpoint='logits',
        in_channels=in_channels,
        dropout_keep_prob=config.dropout_keep_prob,
        sample_duration=config.train_sample_duration)

    if config.checkpoint_path:
        print('Loading pretrained model {}'.format(config.checkpoint_path))
        assert os.path.isfile(config.checkpoint_path)
        pretrained_weights = mindspore.load_checkpoint(config.checkpoint_path)
        mindspore.load_param_into_net(model, pretrained_weights)

        # Setup finetuning layer for different number of classes
        model.replace_logits(config.finetune_num_classes)
        print('Replacing model logits with {} output classes.'.format(config.finetune_num_classes))

        # Setup which layers to train
        parameters_to_train = get_fine_tuning_parameters(model, config.finetune_prefixes)

        return model, parameters_to_train

    return model, model.get_parameters()
