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
"""get input data."""
import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image

def cut(origin_img):
    img_h, img_w = origin_img.size
    if img_h > 480:
        origin_img = origin_img.crop((0, 0, 480, img_w))
    if img_w > 480:
        origin_img = origin_img.crop((0, 0, img_h, 480))
    print("after cut: ", img.size)
    return origin_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/images/train', help='evaling image path')
    parser.add_argument('--output_path', type=str, default='../data/input_cut_train', help='output image path')
    parser.add_argument('--output_noise_path', type=str, default='../data/input_noise_train', help='output image path')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    if not os.path.exists(opt.output_noise_path):
        os.makedirs(opt.output_noise_path)

    # data
    files = glob.glob(opt.dataset_path + '/*')
    print("get input data start!")
    for file in tqdm(files):
        name = file.split("/")[-1]
        print(name)
        img = Image.open(file)
        img = cut(img)
        img.save(os.path.join(opt.output_noise_path, name), format='jpeg', quality=10)
        img.save(os.path.join(opt.output_path, name), format='jpeg', quality=95)
    print("finished!")
