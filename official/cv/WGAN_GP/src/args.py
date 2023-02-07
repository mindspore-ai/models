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

"""get args"""
import argparse

def get_args():
    """Define the common options that are used in training."""
    parser = argparse.ArgumentParser(description='WGAN-GP')
    parser.add_argument('--device_target', default='Ascend', help='enables npu')
    parser.add_argument('--device_id', type=int, default=0)

    parser.add_argument('--dataroot', default=None, help='path to dataset')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height/width of the input image to network')
    parser.add_argument('--DIM', type=int, default=128, help='dimension of input samples')
    parser.add_argument('--niter', type=int, default=1200, help='number of epochs to train for')
    parser.add_argument('--save_iterations', type=int, default=1000, help='num of gen iterations to save model')
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for Critic, default=0.0001')
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam. default=0.9')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--experiment', default="samples", help='Where to store samples and models')
    parser.add_argument('--ckpt_file', default=None, help='path to pretrained ckpt model file')
    parser.add_argument('--output_dir', default=None, help='output path of generated images')

    args_opt = parser.parse_args()
    return args_opt
