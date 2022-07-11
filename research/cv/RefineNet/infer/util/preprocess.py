# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import os
import cv2
import numpy as np

IMAGE_MEAN = [103.53, 116.28, 123.675]
IMAGE_STD = [57.375, 57.120, 58.395]

def _parse_args():
    parser = argparse.ArgumentParser('refinenet eval')
    parser.add_argument('--data_root', type=str, default='', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--output_path', default='../mxbase/bin', type=str, help='bin file path')
    args, _ = parser.parse_known_args()
    return args


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def image_bgr_rgb(img):
    img_data = img[:, :, ::-1]
    return img_data


def img_process(img_, crop_size=513):
    """pre_process"""
    # resize
    img_ = resize_long(img_, crop_size)

    # mean, std
    image_mean = np.array(IMAGE_MEAN)
    image_std = np.array(IMAGE_STD)
    img_ = (img_ - image_mean) / image_std
    img_ = image_bgr_rgb(img_)
    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_


def main():
    args = _parse_args()

    with open(args.data_lst) as f:
        img_lst = f.readlines()
        for i, line in enumerate(img_lst):
            img_path, _ = line.strip().split(' ')
            img_path = os.path.join(args.data_root, img_path)
            image_name = img_path.split(os.sep)[-1].split(".")[0]
            print("The", i+1, "img_path:", img_path)

            # read image
            img = cv2.imread(img_path)

            # preprocess
            img = img_process(img, 513)
            img = np.expand_dims(img, 0)  # NCHW
            img = np.array(img).astype('float32')

            # save bin file
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            data = img
            dataname = image_name + ".bin"
            data.tofile(args.output_path + '/' + dataname)


if __name__ == '__main__':
    main()
