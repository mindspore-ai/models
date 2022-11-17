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
"""train Retinanet."""

import argparse
import os
import subprocess
import time
import moxing as mox
from mindspore import context
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"
_global_sync_count = 0


def _parse_args():
    parser = argparse.ArgumentParser('mindspore retinanet training')

    # dataset
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=81,
                        help='number of classes')
    parser.add_argument('--mindrecord_path', type=str, default='/cache/data/',
                        help='mindrecord path')
    parser.add_argument('--mindrecord_dir', type=str, default='/cache/data/',
                        help='mindrecord_dir path')

    # optimizer
    parser.add_argument('--epoch_size', type=int, default=1, help='epoch')
    parser.add_argument('--enable_modelarts', type=str, default='True',
                        help='enable modelarts')

    # model
    parser.add_argument("--data_path", type=str, default="/cache/data", help="dataset path for local")
    parser.add_argument("--load_path", type=str, default="/cache/checkpoint", help="dataset path for local")
    parser.add_argument("--output_path", type=str, default="/cache/train", help="training output path for local")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="the path where pre-trained checkpoint file path")
    parser.add_argument("--checkpoint_url", type=str, default="",
                        help="the path where pre-trained checkpoint file path")

    # train
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=1, help='save checkpoint epochs')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='checkpoint max value')
    parser.add_argument('--save_checkpoint_path', type=str, default='/cache/train/ckpt', help='ckpt path')
    parser.add_argument('--distribute', type=str, default=False, help='distributed training')
    parser.add_argument('--pre_trained_epoch_size', type=int, default=1, help='pre trained epoch size')
    parser.add_argument('--filter_weight', type=str, default=False,
                        help="filter weight")
    parser.add_argument('--pre_trained', type=str,
                        default="",
                        help="pre_trained")

    # export config
    parser.add_argument('--file_format', type=str, default='AIR', help='file_format')
    parser.add_argument('--file_name', type=str, default='retinanet', help='output air file name')
    parser.add_argument("--device_id", type=int, default=0, help="")

    args, _ = parser.parse_known_args()
    return args


def _train(args, train_url, data_url, pretrained_checkpoint):
    train_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "./train.py")
    cmd = ["python3.7", train_file,
           f"--train_url={os.path.abspath(train_url)}",
           f"--data_url={os.path.abspath(data_url)}",
           f"--enable_modelarts={args.enable_modelarts}",
           f"--batch_size={args.batch_size}",
           f"--num_classes={args.num_classes}",
           f"--pre_trained={args.pre_trained}",
           f"--mindrecord_dir={args.mindrecord_dir}",
           f"--checkpoint_path={pretrained_checkpoint}",
           f"--epoch_size={args.epoch_size}",
           f"--device_target={args.device_target}",
           f"--distribute={args.distribute}",
           f"--pre_trained_epoch_size={args.pre_trained_epoch_size}",
           f"--save_checkpoint_epochs={args.save_checkpoint_epochs}",
           f"--keep_checkpoint_max={args.keep_checkpoint_max}",
           f"--save_checkpoint_path={args.save_checkpoint_path}"]
    if args.distribute:
        cmd.append('--distribute')
    if args.filter_weight == "True":
        cmd.append('--filter_weight=True')
    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()


def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None
    print(os.path.join(ckpt_dir, sorted(ckpt_files)[-1]))
    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _export_air(args, ckpt_dir):
    export_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "./export.py")
    ckpt_file = _get_last_ckpt(ckpt_dir)
    print('checkpoint file path: ', ckpt_file)
    cmd = ["python", export_file,
           f"--batch_size={args.batch_size}",
           f"--checkpoint_path={ckpt_file}",
           f"--num_classes={args.num_classes}",
           f"--file_format={args.file_format}",
           f"--file_name={os.path.join(args.output_path, args.file_name)}"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))


def download_data(args):
    """
    sync data from data_url, train_url to data_path, output_path
    :return:
    """
    if args.enable_modelarts:
        if args.data_url:
            if not os.path.isdir(args.data_path):
                os.makedirs(args.data_path)
                sync_data(args.data_url, args.data_path)
                print("Dataset downloaded: ", os.listdir(args.data_path))
        if args.checkpoint_url:
            if not os.path.isdir(args.load_path):
                os.makedirs(args.load_path)
                sync_data(args.checkpoint_url, args.load_path)
                print("Preload downloaded: ", os.listdir(args.load_path))
        if args.train_url:
            if not os.path.isdir(args.output_path):
                os.makedirs(args.output_path)
                os.makedirs(args.save_checkpoint_path)
            sync_data(args.train_url, args.output_path)
            print("Workspace downloaded: ", os.listdir(args.output_path))

        context.set_context(save_graphs_path=os.path.join(args.output_path, str(get_rank_id())))
        args.device_num = get_device_num()
        args.device_id = get_device_id()
        # create output dir
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)


def upload_data(args):
    """
    sync data from output_path to train_url
    :return:
    """
    if args.enable_modelarts:
        if args.train_url:
            print("Start copy data to output directory.")
            sync_data(args.output_path, args.train_url)
            print("Copy data to output directory finished.")


def main():
    args = _parse_args()
    try:
        train_url = _CACHE_TRAIN_URL
        print("train_url path", train_url)
        data_url = _CACHE_DATA_URL
        download_data(args)
        pretrained_checkpoint = os.path.join(_CACHE_DATA_URL,
                                             args.checkpoint_url) if args.checkpoint_url else ""
        print("pretrained_checkpoint ::::", pretrained_checkpoint)
        ret = _train(args, train_url, data_url, pretrained_checkpoint)
        _export_air(args, args.save_checkpoint_path)
        mox.file.copy_parallel(args.output_path, args.train_url)
        upload_data(args)
    except ModuleNotFoundError:
        train_url = args.train_url
        data_url = args.data_url
        pretrained_checkpoint = args.checkpoint_url
        ret = _train(args, train_url, data_url, pretrained_checkpoint)

    if ret != 0:
        exit(1)


if __name__ == '__main__':
    main()
