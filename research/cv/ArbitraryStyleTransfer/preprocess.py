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
"""preprocess"""
import argparse
import os

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser('preprocess')
parser.add_argument("--content_path", type=str, help='content_path, default: None')
parser.add_argument("--style_path", type=str, help='style_path, default: None')
parser.add_argument("--image_size", type=int, default=256, help='image size, default: image_size.')
parser.add_argument('--output_path', type=str, default="./preprocess_Result", help='eval data dir')
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    image_dir_style = os.path.join(args.output_path, "style")
    if not os.path.exists(image_dir_style):
        os.makedirs(image_dir_style)
    image_dir_content = os.path.join(args.output_path, "content")
    if not os.path.exists(image_dir_content):
        os.makedirs(image_dir_content)
    content_list = os.listdir(args.content_path)
    for j in range(0, len(content_list)):
        pic_path = os.path.join(args.content_path, content_list[j])
        # content_pic = crop_and_resize(pic_path, size=256)
        img_c = Image.open(pic_path).convert("RGB")
        img_c = np.array(img_c.resize((256, 256)))
        img_c = (img_c / 127.5) - 1.0
        img_c = img_c.transpose(2, 0, 1).astype(np.float32)
        img_c = np.reshape(img_c, (1, 3, 256, 256))
        file_name = content_list[j].replace(".jpg", "") + ".bin"
        image_path_content = os.path.join(args.output_path, "content", file_name)
        content_pic_shifted = img_c.copy()
        content_pic_shifted += 1
        content_pic_shifted /= 2
        content_pic_shifted -= np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        content_pic_shifted /= np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        img_c = np.reshape(img_c, (1, 1, 3, 256, 256))
        content_pic_shifted = np.reshape(content_pic_shifted, (1, 1, 3, 256, 256))
        output = np.concatenate((img_c, content_pic_shifted), axis=0)
        output.tofile(image_path_content)

    style_list = os.listdir(args.style_path)

    for j in range(0, len(style_list)):
        pic_path = os.path.join(args.style_path, style_list[j])
        img_c = Image.open(pic_path).convert("RGB")
        img_c = np.array(img_c.resize((256, 256)))
        img_c = (img_c / 127.5) - 1.0
        img_c = img_c.transpose(2, 0, 1).astype(np.float32)
        img_c = np.reshape(img_c, (1, 3, 256, 256))
        style_pic = img_c
        style_pic += 1
        style_pic /= 2
        style_pic -= np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        style_pic /= np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        file_name = style_list[j].replace(".jpg", "") + ".bin"
        image_path_style = os.path.join(args.output_path, "style", file_name)
        style_pic.tofile(image_path_style)
    print("Export bin files finished!")
