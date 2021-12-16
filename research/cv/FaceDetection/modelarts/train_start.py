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
"""train FaceDetection."""

import argparse
import os
import subprocess

# the dataset_train url of modelArts
_CACHE_DATA_URL = "/cache/data_url"
# the output url of modelArts
_CACHE_TRAIN_URL = "/cache/train_url"


def _parse_args():
    """
        Set and return the parameters of the setting
    """
    parser = argparse.ArgumentParser('mindspore FaceDetection training')
    parser.add_argument('--train_url', type=str, default=_CACHE_TRAIN_URL,
                        help='where training log and ckpts saved')

    # dataset_train
    parser.add_argument('--data_url', type=str, default=_CACHE_DATA_URL,
                        help='path of dataset_train')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='number of classes')
    parser.add_argument('--mindrecord_path', type=str, default='/cache/data/data.mindrecord',
                        help='mindrecord path')
    parser.add_argument('--need_modelarts_dataset_unzip', type=str, default='',
                        help='need modelarts dataset_train unzip')

    # optimizer
    parser.add_argument('--max_epoch', type=int, default=10, help='epoch')

    # model
    parser.add_argument('--checkpoint_url', type=str, default='',
                        help='pretrained model')
    parser.add_argument('--ckpt_path', type=str, default='../output',
                        help='the path of saving ckpt')
    parser.add_argument('--pretrained', type=str,
                        default='./ckpt/facedetection_ascend_v120_humanface_research_cv_bs64_acc77.ckpt',
                        help='pretrained ckpt path')

    # train
    parser.add_argument('--run_platform', type=str, default='Ascend',
                        choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--enable_modelarts', type=str, default='True', help='use modelarts')
    parser.add_argument('--file_name', type=str, default='faceDetection', help='output air file name')

    arg, _ = parser.parse_known_args()
    return arg


def _train(arg):
    """"
        Start training model
    """
    train_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "train.py")
    print(train_file)
    cmd = ["python3.7", train_file,
           f"--train_url={os.path.abspath(arg.train_url)}",
           f"--data_url={os.path.abspath(arg.data_url)}",
           f"--checkpoint_url={os.path.abspath(arg.checkpoint_url)}",
           f"--batch_size={arg.batch_size}",
           f"--num_classes={arg.num_classes}",
           f"--max_epoch={arg.max_epoch}",
           f"--pretrained={arg.pretrained}",
           f"--run_platform={arg.run_platform}",
           f"--enable_modelarts={arg.enable_modelarts}",
           f"--mindrecord_path={arg.mindrecord_path}",
           f"--need_modelarts_dataset_unzip={arg.need_modelarts_dataset_unzip}"]

    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)

    return process.wait()


def _get_last_ckpt(arg):
    """
        Result the last ckpt file
        Args:
            arg: the parameters of the setting
        Returns:
            the last ckpt file
    """
    file_dict = {}
    ckpt_dir = os.path.join(arg.train_url, 'output')
    lists = os.listdir(ckpt_dir)
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_dir = os.path.join(ckpt_dir, file_dict[max_ctime])
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _export_air(arg):
    """
        Export the model in AIR format
        Args:
            arg: the parameters of the setting
        Returns:
            null
        Output:
            the model of AIR format
    """
    export_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "export.py")
    ckpt_file = _get_last_ckpt(arg)
    cmd = ["python3.7", export_file,
           f"--batch_size={arg.batch_size}",
           # The pretrained is not the same as the trained in ARG. This is the path of CKPT after train
           f"--pretrained={ckpt_file}",
           f"--num_classes={arg.num_classes}",
           f"--file_name={os.path.join(arg.train_url, arg.file_name)}"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


if __name__ == '__main__':
    args = _parse_args()
    ret = _train(args)
    if ret != 0:
        exit(1)
    _export_air(args)
