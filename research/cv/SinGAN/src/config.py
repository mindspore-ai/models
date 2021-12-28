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
"""Configuration for SinGAN"""
import argparse

def get_arguments():
    """Configurations"""
    parser = argparse.ArgumentParser(description='MindSpore SinGAN')

    #load, input, save configurations:
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training and testing')
    parser.add_argument('--dataset', type=str, default='Photo', help='type of dataset', choices=['Photo'])
    parser.add_argument('--netG', type=str, default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', type=str, default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, default=3, help='noise # channels')
    parser.add_argument('--nc_im', type=int, default=3, help='image # channels')

    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32, help='number of feature channel in conv')
    parser.add_argument('--min_nfc', type=int, default=32, help='minimum number of feature channel in conv')
    parser.add_argument('--ker_size', type=int, default=3, help='kernel size')
    parser.add_argument('--num_layer', type=int, default=5, help='number of conv layers')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--padd_size', type=int, default=0, help='net pad size')

    #pyramid parameters:
    parser.add_argument('--scale_factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min_size', type=int, default=25, help='image minimal size at the coarser scale')
    parser.add_argument('--max_size', type=int, default=250, help='image maximal size at the finer scale')

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='epochs to train per scale, default=2000')
    parser.add_argument('--gamma', type=float, default=0.1, help='scheduler gamma')
    parser.add_argument('--lr_g', type=float, default=5e-4, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=5e-4, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps', type=int, default=3, help='Generator inner steps')
    parser.add_argument('--Dsteps', type=int, default=3, help='Discriminator inner steps')
    parser.add_argument('--lambda_grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--alpha', type=float, default=10, help='reconstruction loss weight')

    return parser
