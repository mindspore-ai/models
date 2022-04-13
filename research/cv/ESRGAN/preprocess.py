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
    preprocess
"""
import os
import argparse
import numpy as np
from src.dataset.testdataset import create_testdataset

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--val_data_dir', type=str, default='', help='eval data dir')
parser.add_argument('--val_batch_size', type=int, default=1, help='batch_size, default is 1.')
parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                    help='during validating, the file path of Generated image.')
args = parser.parse_args()


def padding(_img, target_shape):
    """padding image for 310."""
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = _img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {_img.shape}")
    if dh != 0 or dw != 0:
        _img = np.pad(_img, ((0, dh), (0, dw), (0, 0)), "constant")
    return _img


if __name__ == '__main__':
    result_path = "./preprocess_Result/"
    val_LR_data_dir = os.path.join(args.val_data_dir, 'LRbicx4')
    val_HR_data_dir = os.path.join(args.val_data_dir, 'GTmod12')
    test_psnr_ds = create_testdataset(args.val_batch_size, val_LR_data_dir, val_HR_data_dir)
    img_path = os.path.join(result_path, "00_data")
    os.makedirs(img_path)
    for i, data in enumerate(test_psnr_ds.create_dict_iterator(output_numpy=True)):
        file_name = "ESRGAN_data_bs" + str(args.val_batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        lr = data['LR']
        lr = np.squeeze(lr, axis=0)
        lr = lr.transpose(1, 2, 0)
        org_img = padding(lr, [200, 200])
        org_img = org_img.transpose(2, 0, 1)
        img = org_img.copy()
        img.tofile(file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
