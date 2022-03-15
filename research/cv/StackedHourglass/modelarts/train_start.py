# Copyright 2022 Huawei Technologies Co., Ltd
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
"""train hourglass."""

import argparse
import os
import subprocess


def _parse_args():
    """
    _parse_args
    """
    parser = argparse.ArgumentParser('mindspore hourglass training')
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5)
    parser.add_argument("--keep_checkpoint_max", type=int, default=20)
    parser.add_argument("--initial_lr", type=float, default=0.001)
    parser.add_argument("--decay_rate", type=float, default=0.985)
    parser.add_argument("--decay_epoch", type=int, default=1)
    parser.add_argument("--annot_dir", type=str, default="/cache/data/MPII/annot")
    parser.add_argument("--img_dir", type=str, default="/cache/data/MPII/images")
    parser.add_argument("--file_name", type=str, default="Hourglass")
    parser.add_argument("--file_format", type=str, default="AIR")
    args, _ = parser.parse_known_args()
    return args


parse_args = _parse_args()


def _train():
    """
    _train
    """
    train_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)),
                              "train_modelarts.py")
    cmd = ["python", train_file,
           f"--train_url={parse_args.train_url}",
           f"--data_url={parse_args.data_url}",
           f"--batch_size={parse_args.batch_size}",
           f"--num_epoch={parse_args.num_epoch}",
           f"--save_checkpoint_epochs={parse_args.save_checkpoint_epochs}",
           f"--keep_checkpoint_max={parse_args.keep_checkpoint_max}",
           f"--initial_lr={parse_args.initial_lr}",
           f"--decay_rate={parse_args.decay_rate}",
           f"--decay_epoch={parse_args.decay_epoch}",
           f"--annot_dir={parse_args.annot_dir}",
           f"--img_dir={parse_args.img_dir}",
           f"--file_name={parse_args.file_name}",
           f"--file_format={parse_args.file_format}"]
    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()


def main():
    """
    main
    """
    ret = _train()
    if ret != 0:
        exit(1)


if __name__ == '__main__':
    main()
