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
from os import path as osp
import argparse
import ast
import numpy as np
from src.dataset.dataset import DatasetVal, create_val_dataset
from src.util.utils import save_img, compute_psnr

parser = argparse.ArgumentParser('Postprocess')
parser.add_argument("--val_LR_path", type=str, default=r"Set5/LR")
parser.add_argument("--val_GT_path", type=str, default=r"Set5/HR")
parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                    help='during validating, the file path of Generated image.')
parser.add_argument("--upscale_factor", type=int, default="4", help="scale.")
parser.add_argument('--vgg', type=ast.literal_eval, default=True, help="use vgg")
parser.add_argument('--isgan', type=ast.literal_eval, default=False,
                    help="the param of is_gan decides the way of training ")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                    help='device where the code will be implemented (default: Ascend)')
args = parser.parse_args()

def unpadding(img, target_shape):
    """unpadding image for 310."""
    a, b = target_shape[1], target_shape[2]
    _, img_h, img_w = img.shape
    if img_h > a:
        img = img[:, :a, :]
    if img_w > b:
        img = img[:, :, :b]
    return img


if __name__ == '__main__':
    rst_path = "./result_Files"
    object_imageSize = 800
    dataset = DatasetVal(args.val_GT_path, args.val_LR_path, args)
    ds = create_val_dataset(dataset, args)
    psnr_list = []
    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
        gt = data['target_image']
        gt_img = np.squeeze(gt, axis=0)
        file_name = osp.join(rst_path, "DBPN_data_bs" + str(args.testBatchSize) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float32).reshape(3, object_imageSize, object_imageSize)
        sr_img = unpadding(output, gt_img.shape)
        save_img_path = osp.join('./310_infer_img', 'SR')
        save_img(sr_img, str(i), save_img_path)
        save_img_path = osp.join('./310_infer_img', 'HR')
        save_img(gt_img, str(i), save_img_path)
        cur_psnr = compute_psnr(gt_img, sr_img)
        psnr_list.append(cur_psnr)
        print("===> Processing: {} compute_psnr:{:.4f}.".format(i, cur_psnr))
    psnr_mean = np.mean(psnr_list)
    print("val ending psnr = ", np.mean(psnr_list))
    print("Generate images success!")
