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
import os
import argparse
import subprocess


def args_parser():
    parser = argparse.ArgumentParser(description="train_c3d")
    parser.add_argument('--train_url', type=str,
                        default="obs://mindx-user-5/c3d_modelArts/output/", help='test')
    parser.add_argument('--data_url', type=str,
                        default="obs://mindx-user-5/c3d_modelArts/data/", help='test')
    parser.add_argument('--num_classes', type=str, default='101',
                        help='the number of classes to be 101 or 51')
    parser.add_argument('--batch_size', type=str, default='16')
    parser.add_argument('--epoch', type=str, default='10')
    parser_args, _ = parser.parse_known_args()

    return parser_args


def train(args):
    train_file = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "../train.py")
    config_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "../default_config.yaml")
    cmd = [
        "python",
        train_file,
        f"--config_path={config_path}",
        f"--train_url={args.train_url}",
        f"--data_url={args.data_url}",
        f"--num_classes={args.num_classes}",
        f"--batch_size={args.batch_size}",
        f"--epoch={args.epoch}",
        f"--dataset=UCF101",
        f"--json_path=/cache/data_url/UCF-101_json/",
        f"--img_path=/cache/data_url/UCF-101_img/",
        f"--pre_trained=0",
        f"--sport1m_mean_file_path=/cache/data_url/sport1m_train16_128_mean.npy",
        f"--save_dir=/cache/train_url",
        f"--ckpt_path=/cache/train_url",
        f"--ckpt_file=/cache/train_url",
        f"--mindir_file_name=/cache/train_url/export",
        f"--file_format=AIR",
        f"--is_evalcallback=0"
    ]
    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()


def _get_last_ckpt(ckpt_dir):
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_dir = os.path.join(ckpt_dir, file_dict[max_ctime], 'ckpt_0')
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def export_air(ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return

    export_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../export.py")
    cmd = ["python", export_file,
           f"--ckpt_file={ckpt_file}",
           f"--file_format=AIR",
           f"--num_classes=101",
           f"--batch_size=1",
           f"--mindir_file_name=/cache/train_url/export"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


if __name__ == '__main__':
    _args = args_parser()
    import moxing as mox
    os.makedirs("/cache/train_url", exist_ok=True)
    os.makedirs("/cache/data_url", exist_ok=True)
    mox.file.copy_parallel(_args.data_url, "/cache/data_url")
    ret = os.system('cd /cache/data_url; unzip UCF-101_img.zip')
    if ret == 0:
        print("unzip dataset success")
    ret = train(_args)
    export_air("/cache/train_url")
    mox.file.copy_parallel("/cache/train_url", _args.train_url)
