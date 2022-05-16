# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""train textfusenet."""

import argparse
import os
import moxing as mox


def _parse_args():
    """parse input"""
    parser = argparse.ArgumentParser('mindspore textfusenet training')
    parser.add_argument('--train_url', type=str, default='', help='where training log and ckpts saved')

    # dataset
    parser.add_argument('--data_url', type=str, default='', help='path of dataset')
    # model
    parser.add_argument('--run_distribute', default=False, help='run_distribute')
    # run_distribute
    parser.add_argument('--pre_trained_ckpt', type=str, default='resnet101.ckpt',
                        help='pretrained backbone')
    # epoch_size
    parser.add_argument('--epoch_size', type=int, default=200,
                        help='pretrained backbone')
    # save checkpoint epochs
    parser.add_argument('--save_checkpoint_epochs', type=int, default=50,
                        help='save checkpoint epoch')
    # max epochs
    parser.add_argument('--keep_checkpoint_max', type=int, default=200,
                        help='keep checkpoint max epochs')
    # file format
    parser.add_argument('--file_format', type=str, default="AIR",
                        help='the file format of the export model')

    # coco_root
    parser.add_argument('--coco_root', type=str, default="/cache/data/",
                        help='the path of dataset')
    # mindrecord_dir
    parser.add_argument('--mindrecord_dir', type=str, default="/cache/data/mindrecord/textfusenet",
                        help='the path of mindrecord file')
    # file name
    parser.add_argument('--file_name', type=str, default="/cache/data/air/textfusenet",
                        help='the name of export model')
    # save checkpoint path
    parser.add_argument('--save_checkpoint_path', type=str, default="/cache/data/loss_ckpt/",
                        help='the path to save checkpoint')
    # base step
    parser.add_argument('--base_step', type=int, default=1206,
                        help='base step, the number of images')
    args, _ = parser.parse_known_args()
    return args


def _train(args):
    """train textfusenet"""
    train_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                              "train.py")
    ret = os.system(f'python {train_file} --train_url {args.train_url} --data_url {args.data_url} '
                    f'--pre_trained /cache/data/resnet101_backbone.ckpt --run_distribute {args.run_distribute} '
                    f'--epoch_size {args.epoch_size} --save_checkpoint_epochs {args.save_checkpoint_epochs} '
                    f'--keep_checkpoint_max {args.keep_checkpoint_max} --coco_root {args.coco_root} '
                    f'--mindrecord_dir {args.mindrecord_dir} --save_checkpoint_path {args.save_checkpoint_path} '
                    f'--base_step {args.base_step}')
    return ret


def _export_air(args):
    """export to air"""
    export_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "export.py")
    max_epoch = args.keep_checkpoint_max
    os.system(f'python {export_file} --keep_checkpoint_max {max_epoch} '
              f'--ckpt_file_local /cache/data/loss_ckpt/ckpt_0/text_fuse_net-{max_epoch}_{args.base_step}.ckpt '
              f'--file_name {args.file_name} --file_format {args.file_format}')

def main():
    """mai function"""
    args = _parse_args()
    local_data_url = '/cache/data/'
    pretrained_ckpt_path = '/cache/data/resnet101_backbone.ckpt'
    mox.file.copy_parallel(args.data_url, local_data_url)
    mox.file.copy_parallel(args.pre_trained_ckpt, pretrained_ckpt_path)
    os.system('unzip /cache/data/data.zip -d /cache/data/')
    os.system('mkdir /cache/data/loss_ckpt')
    os.mkdir('/cache/data/ckpt')
    _train(args)
    _export_air(args)
    air_path = '/cache/data/air/'
    ckpt_path = '/cache/data/loss_ckpt/'
    mox.file.copy_parallel(air_path, args.train_url)
    mox.file.copy_parallel(ckpt_path, args.train_url)


if __name__ == '__main__':
    main()
