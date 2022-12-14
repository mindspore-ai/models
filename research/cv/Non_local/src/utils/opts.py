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

# This file was copied from project [kenshohara][3D-ResNets-PyTorch]

"""config"""
import argparse


def parse_opts():
    """arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', default='/root', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='', type=str, help='Directory path of Videos')
    parser.add_argument('--train_data_path', default='', type=str, help='Directory path of train Videos')
    parser.add_argument('--test_data_path', default='', type=str, help='Directory path of test Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')

    parser.add_argument('--dataset', default='kinetics', type=str, help='Used dataset (kinetics)')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. \
                              corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--sample_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=32, type=int, help='Temporal duration of inputs')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--save_checkpoint_epochs', default=1, type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--keep_checkpoint_max', default=2, type=int, help='max saved checkpoints number')
    parser.add_argument('--ckpt', default='', type=str, help='path of saved checkpoint')
    parser.add_argument('--modelarts', type=int, default=0, metavar='N', help='modelarts')
    parser.add_argument('--nl', default=True, type=bool, help='add non-local block')

    parser.add_argument('--device_id', type=int, default=0, metavar='N', help='device-id')

    parser.add_argument('--distributed', type=int, default=0, metavar='N', help='distributed')
    parser.add_argument('--loss_scale', default=1024, type=int, help='loss_scale')
    parser.add_argument('--data_url', type=str, default='', help='dataset file path')
    parser.add_argument('--train_url', type=str, default='', help='train file path')
    parser.add_argument('--device_target', type=str, default='', help='target device')
    parser.add_argument('--mode', type=str, default='multi', help='the mode of inference')
    parser.add_argument('--pretrained_ckpt', type=str, default='resnet50.ckpt', help='the path of pretrained ckpt file')
    args = parser.parse_args()
    return args
