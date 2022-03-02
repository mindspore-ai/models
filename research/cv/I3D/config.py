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
Obtain the parameters required by the training model.
"""

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--video_path', type=str, help='Path to location of dataset videos')
    parser.add_argument('--annotation_path', type=str,
                        help='Path to location of dataset annotation file')
    parser.add_argument('--save_dir', default='./output_standalone/', type=str, help='Where to save training outputs.')

    # Dataset
    parser.add_argument('--mode', type=str, required=True, help='train in rgb mode or flow mode')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset string (ucf101 | hmdb51)')
    parser.add_argument('--num_val_samples', type=int, default=1, help='Number of validation samples for each activity')
    parser.add_argument('--num_classes', default=400, type=int,
                        help='Number of classes (ucf101: 101, hmdb51: 51 if load pretained model set this to 400 '
                             'and change --finetune_num_classes to the dataset '
                             'class num)')

    # Preprocessing pipeline
    parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--train_sample_duration', default=64, type=int,
                        help='Temporal duration of inputs during training ')
    parser.add_argument('--test_sample_duration', default=64, type=int,
                        help='Temporal duration of inputs during testing')

    # Finetuning
    parser.add_argument('--checkpoint_path', default='', type=str, help='Checkpoint file (.pth) of previous training')
    parser.add_argument('--finetune_num_classes', default=51, type=int,
                        help='Number of classes for fine-tuning. set num_classes to 400 when use this.'
                             'set this to dataset class num when use pretrained model')
    parser.add_argument('--finetune_prefixes', default='logits,Mixed_5', type=str,
                        help='Prefixes of layers to finetune, comma seperated')

    # Optimization and Models (general)
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='Dropout keep probability')
    parser.add_argument('--optimizer', default='adam', type=str, help='Which optimizer to use (SGD | adam | rmsprop)')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_de_rate', default=0.5, type=float, help='Rate of learning rate decline')
    parser.add_argument('--lr_de_epochs', default=5, type=int,
                        help='After these epochs, the learning rate will decrease')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight Decay')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch, only relevant for finetuning')
    parser.add_argument('--num_epochs', default=32, type=int, help='Number of epochs to train')

    # Model saving
    parser.add_argument('--checkpoint_frequency', type=int, default=1,
                        help='Save checkpoint after this number of epochs')
    parser.add_argument('--checkpoints_num_keep', type=int, default=16, help='Number of checkpoints to keep')

    # Mindspore
    parser.add_argument('--no_eval', default=False, type=bool, help='Disable evaluation')
    parser.add_argument('--sink_mode', default=True, type=bool, help='Dataset sink mode')
    parser.add_argument('--distributed', default=False, type=bool, help='Distribute train')
    parser.add_argument('--context', default='gr', help='py(pynative) or gr(graph)')
    parser.add_argument('--device_target', default='Ascend', help='Device string')
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--amp_level', default='O3', help='Level for mixed precision training')
    parser.add_argument('--device_num', default=8, type=int, help='Number of device')
    parser.add_argument('--has_back', default=False, type=bool, help='Adjust learning rate')

    # openI
    parser.add_argument('--data_url', help='path to training/inference dataset folder', default='./data')
    parser.add_argument('--train_url', help='model folder to save/load', default='./model')
    parser.add_argument('--openI', default=False, type=bool, help='Train model on openI')

    return parser.parse_args()
