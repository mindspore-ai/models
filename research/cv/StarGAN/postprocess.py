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
"""post process for 310 inference"""
import os
import argparse
import numpy as np
from PIL import Image

def denorm(x):
    image_numpy = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy

def get_img(path):
    data = np.fromfile(path, np.float32)
    data = np.reshape(data, (3, 128, 128))
    data = denorm(data)
    return data

parser = argparse.ArgumentParser(description='PostProcess args')
parser.add_argument('--result_path', type=str, required=True, help='Dataset path')
parser.add_argument('--ori_path', type=str, required=True, help='Train output path')
parser.add_argument('--save_path', type=str, required=True, help='Train output path')


args_opt = parser.parse_args()


if __name__ == '__main__':
    result_path = args_opt.result_path
    ori_path = args_opt.ori_path
    save_path = args_opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ori_img_list = os.listdir(ori_path)
    total_num = int(len(ori_img_list) / 5)
    for i in range(total_num):
        result_list = ()
        ori_file_name = os.path.join(ori_path, 'sop_' + str(i) + '_0.bin')
        ori_img = get_img(ori_file_name)
        print("Start processing", ori_file_name)
        result_list += (ori_img,)
        for j in range(5):
            result_file_name = 'sop_' + str(i) + '_' + str(j) +'_0.bin'
            print("Start processing 310 result", result_file_name)
            result_img = get_img(os.path.join(result_path, result_file_name))
            result_list += (result_img,)
        c = np.concatenate(result_list, axis=1)
        save_img = Image.fromarray(np.uint8(c))
        save_img.save(os.path.join(save_path, str(i)+'.jpg'))
        print("Finish convert file", ori_file_name)
    print("=" * 20, "Convert bin files finished", "=" * 20)
