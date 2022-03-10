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
import numpy as np
from tqdm import tqdm
from PIL import Image


def padding(im, target_shape):
    "padding for 310 infer"
    h, w = target_shape[0], target_shape[1]
    img_h, img_w = im.shape[0], im.shape[1]
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {im.shape}")
    if dh != 0 or dw != 0:
        im = np.pad(im, ((0, int(dh)), (0, int(dw)), (0, 0)), "constant")
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/BSD200', help='evaling image path')
    parser.add_argument('--noise_path', type=str, default='./data/BSD200_310', help='output noise image path')
    parser.add_argument('--output_path', type=str, default='./data/BSD200_jpeg_quality10_310', help='output image path')
    opt = parser.parse_args()

    if not os.path.exists(opt.noise_path):
        os.makedirs(opt.noise_path)

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    # data
    files = glob.glob(opt.dataset_path + '/*')

    for file in tqdm(files):
        name = file.split("/")[-1]
        img = Image.open(file)
        img = np.array(img).astype(np.uint8)
        img = padding(img, target_shape=[480, 480])
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(os.path.join(opt.output_path, name), format='jpeg', quality=95)
        img.save(os.path.join(opt.noise_path, name), format='jpeg', quality=10)
    print("finished!")
