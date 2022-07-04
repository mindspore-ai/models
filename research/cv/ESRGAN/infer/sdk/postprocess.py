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
"""
    postprocess
"""
import os
import argparse
import cv2
import numpy as np
from src.util.util import imwrite, calculate_psnr, get, imfrombytes, img2tensor

parser = argparse.ArgumentParser('postprocess')
parser.add_argument('--HR_path', type=str, default='/Set5/HR', help='HR dir')
parser.add_argument('--predict_result', type=str, default='./infer_result/',
                    help='Result of infer.')
parser.add_argument('--result_path', type=str, default='./result_img/',
                    help='Generated image.')

args = parser.parse_args()


def img_convert(img_np, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert image numpy arrays for 310."""
    result = []
    img_np = np.clip(img_np, *min_max)
    img_np = (img_np - min_max[0]) / (min_max[1] - min_max[0])
    img_np = img_np.transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image
        img_np = np.squeeze(img_np, axis=2)
    else:
        if rgb2bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if out_type == np.uint8:
        # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)
    result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def unpadding(img, target_shape):
    """unpadding image for 310."""
    a, b = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    if img_h > a:
        img = img[:a, :, :]
    if img_w > b:
        img = img[:, :b, :]
    return img

def read_img(image_path):
    img_bytes = get(image_path)
    img_lq = imfrombytes(img_bytes, float32=True)
    img = img2tensor(img_lq, pad=False)
    return img

if __name__ == '__main__':
    rst_path = args.predict_result
    gt_path = args.HR_path
    result_path = args.result_path
    object_imageSize = 800

    psnr_list = []
    for i in range(len(os.listdir(rst_path))):
        if i < 9:
            img_path = os.path.join(gt_path, "img_00" + str(i+1) + "_SRF_4_HR.png")
            file_name = os.path.join(rst_path, "img_00" + str(i+1) + '_SRF_4_LR_0.bin')
        else:
            img_path = os.path.join(gt_path, "img_0" + str(i+1) + "_SRF_4_HR.png")
            file_name = os.path.join(rst_path, "img_0" + str(i+1) + '_SRF_4_LR_0.bin')
        gt = read_img(img_path)
        gt_img = img_convert(gt)
        gt = gt.transpose(1, 2, 0)
        print(file_name)
        output = np.fromfile(file_name, np.float32).reshape(3, object_imageSize, object_imageSize)
        output = output.transpose(1, 2, 0)
        output = unpadding(output, gt.shape)
        output = output.transpose(2, 0, 1)
        sr_img = img_convert(output)
        save_img_path = os.path.join(result_path, f'{i + 1}.png')
        imwrite(sr_img, save_img_path)
        cur_psnr = calculate_psnr(sr_img, gt_img)
        psnr_list.append(cur_psnr)
    psnr_mean = np.mean(psnr_list)
    print("val ending psnr = ", np.mean(psnr_list))
    print("Generate images success!")
