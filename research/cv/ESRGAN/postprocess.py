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
from os import path as osp
import argparse
import cv2
import numpy as np
from src.dataset.testdataset import create_testdataset
from src.util.util import imwrite, calculate_psnr

parser = argparse.ArgumentParser('postprocess')
parser.add_argument('--val_data_dir', type=str, default='/home/stu/ajj/Set14', help='eval data dir')
parser.add_argument('--val_batch_size', type=int, default=1, help='batch_size, default is 1.')
parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                    help='during validating, the file path of Generated image.')
parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                    help='device where the code will be implemented (default: Ascend)')

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


if __name__ == '__main__':
    result_dir = "./result_Files"
    object_imageSize = 800
    rst_path = result_dir

    val_LR_data_dir = os.path.join(args.val_data_dir, 'LRbicx4')
    val_HR_data_dir = os.path.join(args.val_data_dir, 'GTmod12')
    test_psnr_ds = create_testdataset(args.val_batch_size, val_LR_data_dir, val_HR_data_dir)
    psnr_list = []
    for i, data in enumerate(test_psnr_ds.create_dict_iterator(output_numpy=True)):
        gt = data['HR']
        gt = np.squeeze(gt, axis=0)
        gt_img = img_convert(gt)
        gt = gt.transpose(1, 2, 0)
        file_name = os.path.join(rst_path, "ESRGAN_data_bs" + str(args.val_batch_size) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float32).reshape(3, object_imageSize, object_imageSize)
        output = output.transpose(1, 2, 0)
        output = unpadding(output, gt.shape)
        output = output.transpose(2, 0, 1)
        sr_img = img_convert(output)
        save_img_path = osp.join('./310_infer_img', 'sr', f'{i + 1}.png')
        imwrite(sr_img, save_img_path)
        save_img_path = osp.join('./310_infer_img', 'gt', f'{i + 1}.png')
        imwrite(gt_img, save_img_path)
        cur_psnr = calculate_psnr(sr_img, gt_img)
        psnr_list.append(cur_psnr)
    psnr_mean = np.mean(psnr_list)
    print("val ending psnr = ", np.mean(psnr_list))
    print("Generate images success!")
