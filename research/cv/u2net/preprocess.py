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
"""preprocess"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser('preprocess')
parser.add_argument("--content_path", type=str, help='content_path, default: None')
parser.add_argument('--output_path', type=str, default="./preprocess_Result/", help='eval data dir')
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    def normalize(img, im_type):
        """normalize tensor"""
        if im_type == "label":
            return img
        if len(img.shape) == 3:
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        else:
            img = (img - 0.485) / 0.229
        return img


    def crop_and_resize(img_path, im_type, size=320):
        """crop and resize tensors"""
        img = np.array(Image.open(img_path), dtype='float32')
        img = img / 255
        img = normalize(img, im_type)
        h, w = img.shape[:2]
        img = cv2.resize(img, dsize=(0, 0), fx=size / w, fy=size / h)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2).repeat(1, axis=2)
        im = img
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 0, 1)
        im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
        return im


    content_list = os.listdir(args.content_path)

    for j in range(0, len(content_list)):
        pic_path = os.path.join(args.content_path, content_list[j])
        content_pic = crop_and_resize(pic_path, im_type="content", size=320)
        file_name = content_list[j].replace(".jpg", "") + ".bin"
        image_path = os.path.join(args.output_path, file_name)
        content_pic.tofile(image_path)

    print("Export bin files finished!")
