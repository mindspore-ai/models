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
Export I3D mindir model.
"""

import argparse

import numpy as np
import mindspore
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.i3d import InceptionI3D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--file_name', default='i3d_minddir', type=str, help='export file name')
    parser.add_argument('--file_format', default='MINDIR', type=str, help='export file format')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size, Preferably the same as during training')
    parser.add_argument('--device', default='Ascend', help='Device string')
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')

    parser.add_argument('--mode', type=str, required=True, help='rgb, flow')
    parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=64, type=int, help='Temporal duration of inputs during testing')
    parser.add_argument('--num_classes', default=51, type=int, help='Number of classes (ucf101: 101, hmdb51: 51)')
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='Dropout keep probability')

    config = parser.parse_args()
    return config


def run_export():
    config = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device, device_id=config.device_id)

    if config.mode == 'rgb':
        in_channels = 3
    else:
        in_channels = 2
    model = InceptionI3D(
        is_train=False,
        amp_level='O0',
        num_classes=config.num_classes,
        train_spatial_squeeze=False,
        final_endpoint='logits',
        in_channels=in_channels,
        dropout_keep_prob=config.dropout_keep_prob,
        sample_duration=config.sample_duration)
    model.set_train(False)

    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)

    # Usually [config.batch_size, in_channels(flow:2 rgb:3), 256, 224, 224]
    input_data = mindspore.Tensor(
        np.ones([config.batch_size, in_channels, config.sample_duration, config.spatial_size, config.spatial_size]),
        mindspore.float32)
    print('Start export')
    export(model, input_data, file_name=config.file_name, file_format=config.file_format)
    print('Finish export')


if __name__ == '__main__':
    run_export()
