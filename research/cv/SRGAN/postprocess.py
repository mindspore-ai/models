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

"""postprocess"""
import os
import argparse
from PIL import Image
import numpy as np
from src.dataset.testdataset import create_testdataset
from mindspore import context
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
parser = argparse.ArgumentParser(description="SRGAN eval")
parser.add_argument("--test_LR_path", type=str, default='/home/SRGANprofile/Set14/LR')
parser.add_argument("--test_GT_path", type=str, default='/home/SRGANprofile/Set14/HR')
parser.add_argument("--result_path", type=str, default='./result_Files')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--scale", type=int, default=4)
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)

def unpadding(img, target_shape):
    a, b = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    if img_h > a:
        img = img[:a, :, :]
    if img_w > b:
        img = img[:, :b, :]
    return img


if __name__ == '__main__':
    i = 0
    args = parser.parse_args()
    test_ds = create_testdataset(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator(output_numpy=True)
    sr_list = []
    psnr_list = []
    for j in range(0, 12):
        f_name = os.path.join(args.result_path, "SRGAN_data_" + str(j) + "_0.bin")
        sr = np.fromfile(f_name, np.float32).reshape(3, 800, 800)
        sr_list.append(sr)
    for data in test_data_loader:
        lr = data['LR']
        sr = sr_list[i]
        i = i+1
        gt = data['HR']
        bs, c, h, w = lr.shape[:4]
        gt = gt[:, :, : h * 4, : w *4]
        gt = gt[0]
        gt = (gt + 1.0) / 2.0
        gt = gt.transpose(1, 2, 0)
        output = sr.transpose(1, 2, 0)
        output = unpadding(output, gt.shape)
        output = (output + 1.0) / 2.0
        result = Image.fromarray((output * 255.0).astype(np.uint8))
        y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
        y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
        psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
        psnr = float('%.2f' % psnr)
        psnr_list.append(psnr)
    x = np.mean(psnr_list)
    x = float('%.2f' % x)
    print("avg PSNR:", x)
 