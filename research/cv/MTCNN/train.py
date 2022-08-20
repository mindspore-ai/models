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

import argparse
from mindspore.common import set_seed
from src.train_models import train_p_net, train_r_net, train_o_net
import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet/RNet/ONet')
    parser.add_argument('--model', required=True, type=str, choices=['pnet', 'rnet', 'onet'],
                        help="Choose model to train")
    parser.add_argument('--mindrecord_file', dest='mindrecord_file',
                        required=True, help='mindrecord file for training', type=str)
    parser.add_argument('--ckpt_path', dest='ckpt_path', default=cfg.CKPT_DIR,
                        help='save checkpoint directory', type=str)
    parser.add_argument('--save_ckpt_steps', default=1000, type=int, help='steps to save checkpoint')
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=cfg.END_EPOCH, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=cfg.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=cfg.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--device_target', dest='device_target', help='device for training', choices=['GPU', 'Ascend'],
                        default='GPU', type=str)
    parser.add_argument('--distribute', dest='distribute', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    args_ = parser.parse_args()
    return args_

if __name__ == '__main__':
    args = parse_args()
    set_seed(66)
    if args.model == 'pnet':
        train_p_net.train_pnet(args)
    if args.model == 'rnet':
        train_r_net.train_rnet(args)
    if args.model == 'onet':
        train_o_net.train_onet(args)
