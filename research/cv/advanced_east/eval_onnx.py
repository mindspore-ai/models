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
"""Run evaluation for a model exported to ONNX"""

import argparse
import os

import numpy as np
import onnxruntime as ort
from mindspore import Tensor
from PIL import Image
from tqdm import tqdm

from src.config import config as cfg
from src.score import eval_pre_rec_f1


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name

def parse_args():
    """parameters"""
    parser = argparse.ArgumentParser('adveast evaling')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['CPU', 'GPU'],
                        help='device where the code will be implemented. (Default: GPU)')

    parser.add_argument('--onnx_path', type=str, default='AdvancedEast.onnx', help='onnx save location')
    parser.add_argument('--data_dir', type=str, default='./icpr/', help='images and labels save location')
    args_opt = parser.parse_args()

    args_opt.batch_size = 1
    args_opt.train_image_dir_name = args_opt.data_dir + cfg.train_image_dir_name
    args_opt.train_label_dir_name = args_opt.data_dir + cfg.train_label_dir_name
    args_opt.val_fname = cfg.val_fname
    args_opt.max_predict_img_size = cfg.max_predict_img_size

    return args_opt

def eval_score(eval_arg):
    """get network and init"""
    session, input_name = create_session(eval_arg.onnx_path, eval_arg.device_target)
    obj = eval_pre_rec_f1()
    with open(os.path.join(eval_arg.data_dir, eval_arg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()

    img_h, img_w = eval_arg.max_predict_img_size, eval_arg.max_predict_img_size
    x = np.zeros((eval_arg.batch_size, 3, img_h, img_w), dtype=np.float32)
    batch_list = np.arange(0, len(f_list), eval_arg.batch_size)
    for idx in tqdm(batch_list):
        gt_list = []
        for i in range(idx, min(idx + eval_arg.batch_size, len(f_list))):
            item = f_list[i]
            img_filename = str(item).strip().split(',')[0][:-4]
            img_path = os.path.join(eval_arg.train_image_dir_name, img_filename) + '.jpg'

            img = Image.open(img_path)
            img = img.resize((img_w, img_h), Image.NEAREST).convert('RGB')
            img = np.asarray(img)
            img = img / 1.
            mean = np.array((123.68, 116.779, 103.939)).reshape([1, 1, 3])
            img = ((img - mean)).astype(np.float32)
            img = img.transpose((2, 0, 1))
            x[i - idx] = img

            gt_list.append(np.load(os.path.join(eval_arg.train_label_dir_name, img_filename) + '.npy'))
        if idx + eval_arg.batch_size >= len(f_list):
            x = x[:len(f_list) - idx]
        y = Tensor(session.run(None, {input_name: x})[0])
        obj.add(y, gt_list)

    print(obj.val())

if __name__ == '__main__':
    args = parse_args()
    eval_score(args)
