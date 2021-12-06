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
"""Evaluate mIOU and Pixel accuracy"""
import os
import argparse
import ast

import cv2
from PIL import Image
import numpy as np

from src.utils.utils import fast_hist
from build_mindrecord import encode_segmap


def decode_segmap(pred):
    """decode_segmap"""
    mask = np.uint8(pred)

    num_classes = 19
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    rank_classes = range(num_classes)

    class_map = dict(zip(rank_classes, valid_classes))

    for _rank in rank_classes:
        mask[mask == _rank] = class_map[_rank]

    return mask

def get_color(npimg):
    """get_color"""
    cityspallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]
    img = Image.fromarray(npimg.astype('uint8'), "P")
    img.putpalette(cityspallete)
    out_img = np.array(img.convert('RGB'))
    return out_img

def infer(args):
    """infer"""
    images_base = os.path.join(args.dataset_path, 'leftImg8bit/val')
    annotations_base = os.path.join(args.dataset_path, 'gtFine/val')
    hist = np.zeros((args.num_classes, args.num_classes))
    for root, _, files in os.walk(images_base):
        for filename in files:
            if filename.endswith('.png'):
                print("start infer ", filename)
                file_name = filename.split('.')[0]

                prob_file = os.path.join(args.result_path, file_name + "_0.bin")
                flipped_prob_file = os.path.join(args.result_path, file_name + "_flip_0.bin")
                prob = np.fromfile(prob_file, dtype=np.float32)

                prob = prob.reshape(1, 19, 1024, 2048)
                flipped_prob = np.fromfile(flipped_prob_file, dtype=np.float32).reshape(1, 19, 1024, 2048)
                pred = (prob + flipped_prob[:, :, :, ::-1])

                pred = pred.argmax(1).astype(np.uint8)
                folder_name = root.split(os.sep)[-1]

                if args.cal_acc:
                    gtFine_name = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    label_file = os.path.join(annotations_base, folder_name, gtFine_name)
                    label = np.array(cv2.imread(label_file, cv2.IMREAD_GRAYSCALE), np.uint8)
                    label = encode_segmap(label, 255)
                    hist = hist + fast_hist(pred.copy().flatten(), label.flatten(), args.num_classes)

                if args.save_img:
                    # labelIds image
                    predImg_name = filename.replace('leftImg8bit', 'predImg_labelIds')
                    predImg_root = os.path.join(args.output_path, folder_name)
                    predImg_file = os.path.join(predImg_root, predImg_name)
                    if not os.path.isdir(predImg_root):
                        os.makedirs(predImg_root)
                    decode_pred = decode_segmap(pred.copy().squeeze(0))
                    cv2.imwrite(predImg_file, decode_pred, [cv2.IMWRITE_PNG_COMPRESSION])

                    # colorful segmentation image
                    colorImg_name = filename.replace('leftImg8bit', 'predImg_colorful')
                    colorImg_root = args.output_path
                    colorImg_root = os.path.join(colorImg_root.replace('output', 'output_img'), folder_name)
                    colorImg_file = os.path.join(colorImg_root, colorImg_name)
                    if not os.path.isdir(colorImg_root):
                        os.makedirs(colorImg_root)
                    color_pred = get_color(pred.copy().squeeze(0))
                    color_pred = cv2.cvtColor(np.asarray(color_pred), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(colorImg_file, color_pred, [cv2.IMWRITE_PNG_COMPRESSION])

    if args.cal_acc:
        miou = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist) + 1e-10)
        miou = round(np.nanmean(miou) * 100, 2)
        print("mIOU = ", miou, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-DeepLab Inference post-process")
    parser.add_argument("--dataset_path", type=str, default="", help="dataset path for evaluation")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")
    parser.add_argument("--result_path", type=str, default="", help="Prob bin file path.")
    parser.add_argument("--output_path", type=str, default="", help="Output path.")
    parser.add_argument("--save_img", type=ast.literal_eval, default=True, help="Whether save pics after inference.")
    parser.add_argument("--cal_acc", type=ast.literal_eval, default=True, help="Calculate mIOU or not.")
    Args = parser.parse_args()
    infer(Args)
