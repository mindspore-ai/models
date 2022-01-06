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
"""args"""
import argparse
import ast

parser = argparse.ArgumentParser(description='AECRNet')

# Hardware specifications
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/cache/data/',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='RESIDE',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Dense',
                    help='test dataset name')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--patch_size', type=int, default=240,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Training specifications
parser.add_argument('--test_every', type=int, default=4000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--loss_scale', type=float, default=1024.0,
                    help='scaling factor for optim')
parser.add_argument('--init_loss_scale', type=float, default=65536.,
                    help='scaling factor')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# ckpt specifications
parser.add_argument('--ckpt_save_path', type=str, default='./ckpt/',
                    help='path to save ckpt')
parser.add_argument('--ckpt_save_interval', type=int, default=10,
                    help='save ckpt frequency, unit is epoch')
parser.add_argument('--ckpt_save_max', type=int, default=100,
                    help='max number of saved ckpt')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='path of saved ckpt')
parser.add_argument('--filename', type=str, default='')

parser.add_argument('--device_target', type=str, default='GPU')

# ModelArts
parser.add_argument('--modelArts_mode', type=ast.literal_eval, default=False,
                    help='train on modelarts or not, default is False')
parser.add_argument('--data_url', type=str, default='', help='the directory path of saved file')
parser.add_argument('--train_url', type=str, default='', help='')

# CR Loss
parser.add_argument('--neg_num', type=int, default=10)
parser.add_argument('--contra_lambda', type=float, default=0.1, help='weight of contra_loss')
parser.add_argument('--vgg_ckpt_path', type=str, default='./')
parser.add_argument('--vgg_ckpt', type=str, default='vgg19_ImageNet.ckpt', help='filename of vgg checkpoint')


args, unparsed = parser.parse_known_args()

args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
