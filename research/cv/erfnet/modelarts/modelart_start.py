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

import argparse
import os
import subprocess

os.system("pip install torch torchvision")

_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"
_CACHE_PRETRAINED_MODEL_URL = "/cache/pretrained_model"

def _parse_args():
    parser = argparse.ArgumentParser('mindspore yolov4_tiny training')
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    parser.add_argument('--pretrained_model_url', type=str, default='')
    parser.add_argument('--num_calss', type=int)
    parser.add_argument('--epoch', type=int)
    args, _ = parser.parse_known_args()
    return args


def _train(args, train_url, data_url, pretrained_model_url):
    train_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "train.py")
    num_calss = args.num_calss
    epoch = args.epoch
    cmd = ["python", train_file,
           f"--lr=5e-4",
           f"--repeat=1",
           f"--run_distribute=false",
           f"--save_path={train_url}",
           f"--num_class={num_calss}",
           f"--data_path={data_url}",
           f"--epoch={epoch}",
           f"--ckpt_path={pretrained_model_url}",
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
    ckpt_file_path = os.path.join(ckpt_dir, file_dict[max_ctime])
    return ckpt_file_path


def _export_air(args, ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return
    import numpy as np
    from mindspore import Tensor, context, load_checkpoint, export
    from src.model import ERFNet
    net = ERFNet(args.num_calss, "XavierUniform",
                 run_distribute=False, train=False)
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target="Ascend")
    context.set_context(device_id=0)
    load_checkpoint(ckpt_file, net=net)
    net.set_train(False)
    input_data = Tensor(np.zeros([1, 3, 512, 1024]).astype(np.float32))
    export(net, input_data, file_name=os.path.join(
        _CACHE_TRAIN_URL, "ERFNet"), file_format="AIR")

def main():

    args = _parse_args()
    import moxing as mox
    os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
    os.makedirs(_CACHE_DATA_URL, exist_ok=True)
    mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
    mox.file.copy_parallel(args.pretrained_model_url,
                           _CACHE_PRETRAINED_MODEL_URL)

    train_url = _CACHE_TRAIN_URL
    data_url = _CACHE_DATA_URL
    if not os.listdir(_CACHE_PRETRAINED_MODEL_URL):
        raise RuntimeError("no model file!")
    pretrained_model_url = os.path.join(
        _CACHE_PRETRAINED_MODEL_URL, os.listdir(_CACHE_PRETRAINED_MODEL_URL)[0])
    ret = _train(args, train_url, data_url, pretrained_model_url)
    _export_air(args, train_url)

    mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    if ret != 0:
        exit(1)

if __name__ == '__main__':
    main()
