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

"""config param"""

import argparse
import ast


def get_args(is_gan=False):
    """return args"""
    parser = argparse.ArgumentParser(description="DBPN train")
    # data resource configuration
    parser.add_argument("--train_GT_path", type=str, default=r'/data/DBPN_data/DIV2K_train_HR')
    parser.add_argument("--val_LR_path", type=str, default=r'/data/DBPN_data/Set5/LR')
    parser.add_argument("--val_GT_path", type=str, default=r'/data/DBPN_data/Set5/HR')
    parser.add_argument("--valDataset", type=str, default="Set5", choices=["Set5", "Set14"], help="eval dataset type")
    parser.add_argument('--eval_flag', type=ast.literal_eval, default=True,
                        help="The flag means whether to eval while training")
    parser.add_argument('--load_pretrained', type=ast.literal_eval, default=False, help='it means whether load ckpt')
    parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8],
                        help="Super resolution upscale factor")
    parser.add_argument('--snapshots', type=int, default=25, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--model_type', type=str, default='DDBPN', choices=["DBPNS", "DDBPN", "DBPN", "DDBPNL"])
    parser.add_argument('--data_augmentation', type=ast.literal_eval, default=True, help="use data augmentation")
    parser.add_argument('--vgg', type=ast.literal_eval, default=True, help="use vgg")
    parser.add_argument('--isgan', type=ast.literal_eval, default=is_gan, help="is_gan decides the way of training")
    # eval setting
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    # distribute
    parser.add_argument("--run_distribute", type=int, default=0, help="run distribute, default: false.")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'),
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 0.")
    parser.add_argument("--rank", type=int, default=0, help="rank id, default: 0.")
    # additional parameters
    parser.add_argument('--sens', type=float, default=1024.0)
    if is_gan:
        parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
        parser.add_argument('--batchSize', type=int, default=4, choices=[4, 16], help='training batch size')
        parser.add_argument('--patch_size', type=int, default=60, choices=[40, 60], help='Size of cropped HR image')
        parser.add_argument('--pretrained_iter', type=int, default=100, help='number of epochs to train for')
        parser.add_argument('--pretrained_D', default='discirminator.pth', help='Sr pretrained base model')
        parser.add_argument('--load_pretrained_D', type=ast.literal_eval, default=False, help='load discriminator flag')
        parser.add_argument('--pretrained', type=ast.literal_eval, default=False)
        parser.add_argument('--load_pretrained_G', type=ast.literal_eval, default=True, help='load generator flag')
        parser.add_argument('--save_folder', default='ckpt/gan', help='Location to save checkpoint models')
        parser.add_argument('--pretrained_dbpn', default='dbpn_100.ckpt', help='sr pretrained base model')
        parser.add_argument('--Results', default='Results/gan', help='eval image')
    else:
        parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
        parser.add_argument('--batchSize', type=int, default=16, choices=[4, 16], help='training batch size')
        parser.add_argument('--patch_size', type=int, default=40, choices=[40, 60], help='Size of cropped HR image')
        parser.add_argument('--save_folder', default='ckpt/gen', help='Location to save checkpoint models')
        parser.add_argument('--pretrained_sr', default='pretrained.ckpt', help='sr pretrained base model')
        parser.add_argument('--Results', default='Results/gen/', help='eval image')
    args = parser.parse_args()
    return args
