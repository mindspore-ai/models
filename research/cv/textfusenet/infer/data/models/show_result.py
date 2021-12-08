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
"""show the result."""

import argparse
import os
import cv2


def parse_args():
    """parse input"""
    parser = argparse.ArgumentParser("show detection result")
    parser.add_argument("--img_path", type=str, required=True, help="image path")
    parser.add_argument("--detect_path", type=str, required=True, help="detection path")
    parser.add_argument("--save_path", type=str, required=True, help="save path")
    args_ = parser.parse_args()
    return args_

def show_result(img_path, detect_path, save_path):
    """show the detection result"""
    img = cv2.imread(img_path)
    f = open(detect_path)
    for line in f:
        line = line[:len(line)-1]
        x_y = line.split(',')
        length = len(x_y)
        for j in range(0, len(x_y)-2, 2):
            cv2.line(img, (int(x_y[j]), int(x_y[j+1])), (int(x_y[j+2]), int(x_y[j+3])), color=(0, 0, 255), thickness=2)
        cv2.line(img, (int(x_y[length-4]), int(x_y[length-3])), (int(x_y[length-2]), int(x_y[length-1])),
                 color=(0, 0, 255), thickness=2)
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    args = parse_args()
    img_list = os.listdir(args.img_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for img_name in img_list:
        show_result(args.img_path + "/" + img_name, args.detect_path + "/" + img_name.replace(".jpg", ".txt"),
                    args.save_path + "/" + img_name)
