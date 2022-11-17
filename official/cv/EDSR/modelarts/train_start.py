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
"""train edsr om modelarts"""
import argparse
import os
import subprocess
import moxing as mox


_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"

def _parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='train and export edsr on modelarts')
    # train output path
    parser.add_argument('--train_url', type=str, default='', help='where training log and ckpts saved')
    # dataset dir
    parser.add_argument('--data_url', type=str, default='', help='where training log and ckpts saved')
    # train config
    parser.add_argument('--data_train', type=str, default='DIV2K', help='train dataset name')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--init_loss_scale', type=float, default=65536., help='scaling factor')
    parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss_scale')
    parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
    parser.add_argument('--ckpt_save_path', type=str, default='ckpt', help='path to save ckpt')
    parser.add_argument('--ckpt_save_interval', type=int, default=10, help='save ckpt frequency, unit is epoch')
    parser.add_argument('--ckpt_save_max', type=int, default=5, help='max number of saved ckpt')
    parser.add_argument('--task_id', type=int, default=0)
    # export config
    parser.add_argument("--export_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--export_file_name", type=str, default="edsr", help="output file name.")
    parser.add_argument("--export_file_format", type=str, default="AIR",
                        choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
    args, _ = parser.parse_known_args()

    return args


def _train(args, data_url):
    """use train.py"""
    pwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    train_file = os.path.join(pwd, "train.py")

    cmd = ["python", train_file,
           f"--dir_data={os.path.abspath(data_url)}",
           f"--data_train={args.data_train}",
           f"--epochs={args.epochs}",
           f"--batch_size={args.batch_size}",
           f"--lr={args.lr}",
           f"--init_loss_scale={args.init_loss_scale}",
           f"--loss_scale={args.loss_scale}",
           f"--scale={args.scale}",
           f"--task_id={args.task_id}",
           f"--ckpt_save_path={os.path.join(_CACHE_TRAIN_URL,args.ckpt_save_path)}",
           f"--ckpt_save_interval={args.ckpt_save_interval}",
           f"--ckpt_save_max={args.ckpt_save_max}"]

    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()

def _get_last_ckpt(ckpt_dir):
    """get the last ckpt path"""
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    if not lists:
        print("No ckpt file found.")
        return None
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_file = os.path.join(ckpt_dir, file_dict[max_ctime])

    return ckpt_file



def _export_air(args, ckpt_dir):
    """export"""
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return
    pwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    export_file = os.path.join(pwd, "export.py")
    cmd = ["python", export_file,
           f"--batch_size={args.export_batch_size}",
           f"--ckpt_path={ckpt_file}",
           f"--file_name={os.path.join(_CACHE_TRAIN_URL, args.export_file_name)}",
           f"--file_format={args.export_file_format}",]
    print(f"Start exporting, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def main():
    args = _parse_args()

    os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
    os.makedirs(_CACHE_DATA_URL, exist_ok=True)

    mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
    data_url = _CACHE_DATA_URL

    _train(args, data_url)
    _export_air(args, os.path.join(_CACHE_TRAIN_URL, args.ckpt_save_path))
    mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)



if __name__ == '__main__':
    main()
