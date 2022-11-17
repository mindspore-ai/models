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
"""train FCN8s."""

import argparse
import os
import subprocess

_CACHE_DATA_URL = "./cache/data"
_CACHE_TRAIN_URL = "./cache/train"



def _parse_args():
    parser = argparse.ArgumentParser('mindspore FCN8s training')

    # url for modelarts
    parser.add_argument('--data_url', type=str, default='',
                        help='Url for modelarts')
    parser.add_argument('--train_url', type=str, default='',
                        help='Url for modelarts')

    # dataset
    parser.add_argument('--crop_size', type=int, default=512, help='crop_size')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes')
    parser.add_argument('--model', type=str, default='FCN8s', help='select model')
    parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size')
    parser.add_argument('--min_scale', type=float, default=0.5, help='min scales of train')
    parser.add_argument('--max_scale', type=float, default=2.0, help='max scales of train')
    parser.add_argument('--data_file', type=str,
                        default='vocaug_mindrecords/voctrain.mindrecord0',
                        help='path of mindrecords')

    # optimizer
    parser.add_argument('--train_epochs', type=int, default=500, help='train epoch')
    parser.add_argument('--base_lr', type=float, default=0.015, help='base lr')
    parser.add_argument('--loss_scale', type=float, default=1024, help='loss scales')

    # model
    parser.add_argument('--ckpt_vgg16', type=str, default='', help='backbone pretrain')
    parser.add_argument('--ckpt_pre_trained', type=str, default='',
                        help='model pretrain')
    parser.add_argument('--save_steps', type=int, default=330, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=5, help='max checkpoint for saving')
    parser.add_argument('--ckpt_dir', type=str, default='', help='where ckpts saved')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'GPU'],
                        help='device id of GPU or Ascend. (Default: Ascend)')
    parser.add_argument('--file_name', type=str, default='fcn8s', help='export file name')
    parser.add_argument('--file_format', type=str, default="AIR",
                        choices=['AIR', 'MINDIR'],
                        help='export model type')
    parser.add_argument('--filter_weight', type=str, default=False, help="filter weight")
    args, _ = parser.parse_known_args()
    return args

def _train(args, train_url, data_url, ckpt_vgg16, ckpt_pre_trained):
    train_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                              "train.py")
    data_file = os.path.join(data_url, args.data_file)
    cmd = ["python", train_file,
           f"--output_path={os.path.abspath(train_url)}",
           f"--data_path={os.path.abspath(data_url)}",
           f"--crop_size={args.crop_size}",
           f"--ignore_label={args.ignore_label}",
           f"--num_classes={args.num_classes}",
           f"--model={args.model}",
           f"--train_batch_size={args.train_batch_size}",
           f"--min_scale={args.min_scale}",
           f"--max_scale={args.max_scale}",
           f"--data_file={data_file}",
           f"--train_epochs={args.train_epochs}",
           f"--base_lr={args.base_lr}",
           f"--loss_scale={args.loss_scale}",
           f"--ckpt_vgg16={ckpt_vgg16}",
           f"--ckpt_pre_trained={ckpt_pre_trained}",
           f"--save_steps={args.save_steps}",
           f"--keep_checkpoint_max={args.keep_checkpoint_max}",
           f"--ckpt_dir={os.path.abspath(train_url)}",
           f"--filter_weight={args.filter_weight}",
           f"--device_target={args.device_target}"]
    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()

def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])

def _export_air(args, ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return

    export_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                               "export.py")
    file_name = os.path.join(ckpt_dir, args.file_name)
    cmd = ["python", export_file,
           f"--file_format={args.file_format}",
           f"--ckpt_file={ckpt_file}",
           f"--num_classes={args.num_classes}",
           f"--file_name={file_name}",
           f"--device_target={args.device_target}"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()

def main():
    args = _parse_args()
    try:
        import moxing as mox
        os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
        os.makedirs(_CACHE_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
        data_url = _CACHE_DATA_URL
        train_url = _CACHE_TRAIN_URL
        ckpt_vgg16 = os.path.join(data_url, args.ckpt_vgg16) \
            if args.ckpt_vgg16 else ""
        ckpt_pre_trained = os.path.join(data_url, args.ckpt_pre_trained) \
            if args.ckpt_pre_trained else ""
        ret = _train(args, train_url, data_url, ckpt_vgg16, ckpt_pre_trained)
        _export_air(args, train_url)
        mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    except ModuleNotFoundError:
        train_url = args.train_url
        data_url = args.data_url
        ckpt_pre_trained = args.ckpt_pre_trained
        ret = _train(args, train_url, data_url, ckpt_vgg16, ckpt_pre_trained)
        _export_air(args, train_url)

    if ret != 0:
        exit(1)

if __name__ == '__main__':
    main()
